use async_trait::async_trait;
use nebula_core::{
    context::BakeContext, error::NebulaError, progress::ProgressReporter,
    scene::SceneGeometry, traits::BakePass,
};
use nebula_gpu::texture::{BakeTexture, TextureFormat2D};
use crate::{config::LightmapConfig, output::{AtlasRegion, LightmapOutput}};

/// GPU path-traced lightmap baker.
///
/// Construct with `LightmapBaker::default()`, then call [`BakePass::execute`].
#[derive(Default)]
pub struct LightmapBaker;

#[async_trait]
impl BakePass for LightmapBaker {
    type Input  = LightmapConfig;
    type Output = LightmapOutput;

    fn name(&self) -> &'static str { "lightmap" }

    async fn execute(
        &self,
        scene:    &SceneGeometry,
        config:   &LightmapConfig,
        ctx:      &BakeContext,
        reporter: &dyn ProgressReporter,
    ) -> Result<LightmapOutput, NebulaError> {
        let res = config.resolution.clamp(64, nebula_gpu::MAX_TEXTURE_DIM);
        reporter.begin("lightmap", 4);

        // ── Step 1: upload scene geometry to GPU ──────────────────────────
        reporter.step("lightmap", 0, "uploading scene geometry");
        let (vertex_buf, index_buf, mesh_info_buf) = upload_scene_geometry(scene, ctx)?;

        // ── Step 2: upload lights ─────────────────────────────────────────
        reporter.step("lightmap", 1, "uploading light data");
        let light_buf = upload_lights(scene, ctx)?;

        // ── Step 3: dispatch bake compute passes ──────────────────────────
        reporter.step("lightmap", 2, &format!(
            "baking {}×{} ({}spp, {} bounces)",
            res, res, config.samples_per_texel, config.bounce_count
        ));
        let format = if config.hdr_output {
            TextureFormat2D::RGBA32F
        } else {
            TextureFormat2D::RGBA16F
        };
        let lightmap = BakeTexture::new(
            &ctx.device, "nebula_lightmap", res, res, format, 1,
            wgpu::TextureUsages::empty(),
        );
        dispatch_lightmap_passes(
            scene, config, ctx, &lightmap,
            &vertex_buf, &index_buf, &mesh_info_buf, &light_buf,
        )?;

        // ── Step 4: readback ──────────────────────────────────────────────
        reporter.step("lightmap", 3, "reading back texels");
        let texels = lightmap.read_back(&ctx.device, &ctx.queue);

        let atlas_regions = build_atlas_regions(scene, res);
        let config_json = serde_json::to_string(config)
            .unwrap_or_else(|_| "{}".to_owned());

        reporter.finish("lightmap", true, &format!("{} texels written", texels.len()));

        Ok(LightmapOutput {
            width: res, height: res,
            channels: 4,
            is_f32: config.hdr_output,
            texels,
            atlas_regions,
            config_json,
        })
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

fn upload_scene_geometry(
    scene: &SceneGeometry,
    ctx:   &BakeContext,
) -> Result<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer), NebulaError> {
    use wgpu::util::DeviceExt;
    use bytemuck::{Pod, Zeroable};

    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    struct GpuVertex {
        pos:    [f32; 3],
        _pad0:  f32,
        normal: [f32; 3],
        _pad1:  f32,
        uv:     [f32; 2],
        lm_uv:  [f32; 2],
    }

    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    struct GpuMeshInfo {
        index_offset:  u32,
        index_count:   u32,
        vertex_offset: u32,
        material_id:   u32,
        transform:     [[f32; 4]; 4],
    }

    let mut vertices: Vec<GpuVertex>   = Vec::new();
    let mut indices:  Vec<u32>         = Vec::new();
    let mut mesh_info: Vec<GpuMeshInfo> = Vec::new();

    for (mesh_idx, mesh) in scene.meshes.iter().enumerate() {
        let vert_base = vertices.len() as u32;
        let idx_base  = indices.len()  as u32;

        let fallback_lm: Vec<[f32; 2]> = mesh.uvs.clone();
        let lm_uvs = mesh.lightmap_uvs.as_ref().unwrap_or(&fallback_lm);

        for i in 0..mesh.positions.len() {
            vertices.push(GpuVertex {
                pos:    mesh.positions[i],
                _pad0:  0.0,
                normal: mesh.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
                _pad1:  0.0,
                uv:     mesh.uvs.get(i).copied().unwrap_or([0.0; 2]),
                lm_uv:  lm_uvs.get(i).copied().unwrap_or([0.0; 2]),
            });
        }
        indices.extend(mesh.indices.iter().map(|&i| i + vert_base));

        let mat = mesh.world_transform.0.to_cols_array_2d();
        mesh_info.push(GpuMeshInfo {
            index_offset:  idx_base,
            index_count:   mesh.indices.len() as u32,
            vertex_offset: vert_base,
            material_id:   mesh.material_ids.first().copied().unwrap_or(0),
            transform:     mat,
        });
    }

    let vbuf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("nebula_lm_vbuf"),
        contents: bytemuck::cast_slice(&vertices),
        usage:    wgpu::BufferUsages::STORAGE,
    });
    let ibuf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("nebula_lm_ibuf"),
        contents: bytemuck::cast_slice(&indices),
        usage:    wgpu::BufferUsages::STORAGE,
    });
    let mbuf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("nebula_lm_mbuf"),
        contents: bytemuck::cast_slice(&mesh_info),
        usage:    wgpu::BufferUsages::STORAGE,
    });
    Ok((vbuf, ibuf, mbuf))
}

fn upload_lights(
    scene: &SceneGeometry,
    ctx:   &BakeContext,
) -> Result<wgpu::Buffer, NebulaError> {
    use wgpu::util::DeviceExt;
    use bytemuck::{Pod, Zeroable};
    use nebula_core::scene::LightSourceKind;

    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    struct GpuLight {
        pos_range:        [f32; 4],
        dir_outer:        [f32; 4],
        color_intensity:  [f32; 4],
        kind:             u32,
        inner_angle:      f32,
        _pad:             [u32; 2],
    }

    let gpu_lights: Vec<GpuLight> = scene.lights.iter().filter(|l| l.bake_enabled).map(|l| {
        match &l.kind {
            LightSourceKind::Directional { direction } => GpuLight {
                pos_range:       [0.0; 4],
                dir_outer:       [direction[0], direction[1], direction[2], 0.0],
                color_intensity: [l.color[0], l.color[1], l.color[2], l.intensity],
                kind: 0, inner_angle: 0.0, _pad: [0; 2],
            },
            LightSourceKind::Point { position, range } => GpuLight {
                pos_range:       [position[0], position[1], position[2], *range],
                dir_outer:       [0.0; 4],
                color_intensity: [l.color[0], l.color[1], l.color[2], l.intensity],
                kind: 1, inner_angle: 0.0, _pad: [0; 2],
            },
            LightSourceKind::Spot { position, direction, range, inner_angle, outer_angle } => GpuLight {
                pos_range:       [position[0], position[1], position[2], *range],
                dir_outer:       [direction[0], direction[1], direction[2], outer_angle.cos()],
                color_intensity: [l.color[0], l.color[1], l.color[2], l.intensity],
                kind: 2, inner_angle: inner_angle.cos(), _pad: [0; 2],
            },
            LightSourceKind::Area { center, right, up, half_w, half_h } => GpuLight {
                pos_range:       [center[0], center[1], center[2], half_w * half_h],
                dir_outer:       [right[0], right[1], right[2], 0.0],
                color_intensity: [l.color[0], l.color[1], l.color[2], l.intensity],
                kind: 3, inner_angle: 0.0, _pad: [0; 2],
            },
            _ => GpuLight { pos_range: [0.0;4], dir_outer: [0.0;4], color_intensity: [0.0;4], kind: 0, inner_angle: 0.0, _pad: [0;2] },
        }
    }).collect();

    let data: &[u8] = if gpu_lights.is_empty() {
        &[0u8; 48] // at least one element so the buffer is valid
    } else {
        bytemuck::cast_slice(&gpu_lights)
    };

    Ok(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("nebula_lm_lights"),
        contents: data,
        usage:    wgpu::BufferUsages::STORAGE,
    }))
}

fn dispatch_lightmap_passes(
    scene:      &SceneGeometry,
    config:     &LightmapConfig,
    ctx:        &BakeContext,
    lightmap:   &nebula_gpu::texture::BakeTexture,
    vbuf:       &wgpu::Buffer,
    ibuf:       &wgpu::Buffer,
    mbuf:       &wgpu::Buffer,
    light_buf:  &wgpu::Buffer,
) -> Result<(), NebulaError> {
    use nebula_gpu::WORKGROUP_SIZE;
    use bytemuck::{Pod, Zeroable};

    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    struct LmParams {
        resolution:        u32,
        samples_per_texel: u32,
        bounce_count:      u32,
        num_lights:        u32,
        num_meshes:        u32,
        num_indices:       u32,
        max_ray_dist:      f32,
        frame_seed:        u32,
    }

    let total_indices: usize = scene.meshes.iter().map(|m| m.indices.len()).sum();
    let params = LmParams {
        resolution:        config.resolution,
        samples_per_texel: config.samples_per_texel,
        bounce_count:      config.bounce_count,
        num_lights:        scene.lights.iter().filter(|l| l.bake_enabled).count() as u32,
        num_meshes:        scene.meshes.len() as u32,
        num_indices:       total_indices as u32,
        max_ray_dist:      config.max_ray_distance,
        frame_seed:        0x12345678,
    };

    use wgpu::util::DeviceExt;
    let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("nebula_lm_params"),
        contents: bytemuck::bytes_of(&params),
        usage:    wgpu::BufferUsages::UNIFORM,
    });

    let bind_group_layout = ctx.device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label: Some("nebula_lm_bgl"),
            entries: &[
                // 0: params uniform
                bgl_entry(0, wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None,
                }, wgpu::ShaderStages::COMPUTE),
                // 1: vertices
                bgl_entry(1, storage_ro(), wgpu::ShaderStages::COMPUTE),
                // 2: indices
                bgl_entry(2, storage_ro(), wgpu::ShaderStages::COMPUTE),
                // 3: mesh info
                bgl_entry(3, storage_ro(), wgpu::ShaderStages::COMPUTE),
                // 4: lights
                bgl_entry(4, storage_ro(), wgpu::ShaderStages::COMPUTE),
                // 5: output lightmap (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access:         wgpu::StorageTextureAccess::WriteOnly,
                        format:         lightmap.format,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        }
    );

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:  Some("nebula_lm_bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: vbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: ibuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: mbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: light_buf.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding:  5,
                resource: wgpu::BindingResource::TextureView(&lightmap.view),
            },
        ],
    });

    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:  Some("nebula_lightmap_cs"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(LIGHTMAP_WGSL)),
    });

    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label:                Some("nebula_lm_pipeline_layout"),
        bind_group_layouts:   &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label:       Some("nebula_lightmap_pipeline"),
        layout:      Some(&pipeline_layout),
        module:      &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut enc = ctx.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("nebula_lm_enc") }
    );
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("nebula_lightmap_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let wg = WORKGROUP_SIZE;
        let res = config.resolution;
        pass.dispatch_workgroups(res.div_ceil(wg), res.div_ceil(wg), 1);
    }
    ctx.queue.submit(std::iter::once(enc.finish()));
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

    Ok(())
}

fn build_atlas_regions(scene: &SceneGeometry, resolution: u32) -> Vec<AtlasRegion> {
    // Simple equal-area tiling: N meshes → ceil(sqrt(N)) × ceil(sqrt(N)) grid.
    let n = scene.meshes.len();
    if n == 0 { return Vec::new(); }
    let cols = (n as f64).sqrt().ceil() as u32;
    let rows = (n as u32).div_ceil(cols);
    let cell_w = 1.0 / cols as f32;
    let cell_h = 1.0 / rows as f32;

    scene.meshes.iter().enumerate().map(|(i, mesh)| {
        let col = (i as u32) % cols;
        let row = (i as u32) / cols;
        AtlasRegion {
            mesh_id:   mesh.id,
            uv_offset: [col as f32 * cell_w, row as f32 * cell_h],
            uv_scale:  [cell_w, cell_h],
        }
    }).collect()
}

// ── Tiny helpers ──────────────────────────────────────────────────────────────

fn bgl_entry(idx: u32, ty: wgpu::BindingType, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: idx, visibility: vis, ty, count: None }
}

fn storage_ro() -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size:   None,
    }
}

// ── Lightmap compute shader ───────────────────────────────────────────────────

const LIGHTMAP_WGSL: &str = r#"
// ── Nebula lightmap baker — GPU path tracer ──────────────────────────────────
//
// This is a reference Monte Carlo path tracer.  It is intentionally readable
// over micro-optimised — production quality comes from more samples, not from
// shader tricks.  Replace with a hardware ray-tracing pipeline on devices that
// support it.

struct Params {
    resolution:        u32,
    samples_per_texel: u32,
    bounce_count:      u32,
    num_lights:        u32,
    num_meshes:        u32,
    num_indices:       u32,
    max_ray_dist:      f32,
    frame_seed:        u32,
}

struct Vertex {
    pos:    vec3<f32>,
    _pad0:  f32,
    normal: vec3<f32>,
    _pad1:  f32,
    uv:     vec2<f32>,
    lm_uv:  vec2<f32>,
}

struct MeshInfo {
    index_offset:  u32,
    index_count:   u32,
    vertex_offset: u32,
    material_id:   u32,
    transform:     mat4x4<f32>,
}

struct GpuLight {
    pos_range:       vec4<f32>,
    dir_outer:       vec4<f32>,
    color_intensity: vec4<f32>,
    kind:            u32,
    inner_angle:     f32,
    _pad:            vec2<u32>,
}

@group(0) @binding(0) var<uniform>         params:     Params;
@group(0) @binding(1) var<storage, read>   vertices:   array<Vertex>;
@group(0) @binding(2) var<storage, read>   indices:    array<u32>;
@group(0) @binding(3) var<storage, read>   meshes:     array<MeshInfo>;
@group(0) @binding(4) var<storage, read>   lights:     array<GpuLight>;
@group(0) @binding(5) var                  out_lm:     texture_storage_2d<rgba32float, write>;

// ── PCG random ───────────────────────────────────────────────────────────────

var<private> rng_state: u32;

fn pcg_next() -> u32 {
    rng_state = rng_state * 747796405u + 2891336453u;
    let word = ((rng_state >> ((rng_state >> 28u) + 4u)) ^ rng_state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f() -> f32 { return f32(pcg_next()) * (1.0 / 4294967296.0); }

fn rand_hemisphere(n: vec3<f32>) -> vec3<f32> {
    let u1 = rand_f();
    let u2 = rand_f();
    let r   = sqrt(1.0 - u1 * u1);
    let phi = 6.28318530718 * u2;
    // Build local ONB
    var t  = vec3<f32>(1.0, 0.0, 0.0);
    if abs(n.x) > 0.9 { t = vec3<f32>(0.0, 1.0, 0.0); }
    let b  = normalize(cross(n, t));
    let tt = cross(b, n);
    return normalize(r * cos(phi) * tt + r * sin(phi) * b + u1 * n);
}

// ── Möller–Trumbore ray–triangle intersection ─────────────────────────────────

fn ray_triangle(
    ro: vec3<f32>, rd: vec3<f32>,
    v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>,
) -> f32 {
    let e1  = v1 - v0;
    let e2  = v2 - v0;
    let h   = cross(rd, e2);
    let det = dot(e1, h);
    if abs(det) < 1e-7 { return -1.0; }
    let inv = 1.0 / det;
    let s   = ro - v0;
    let u   = inv * dot(s, h);
    if u < 0.0 || u > 1.0 { return -1.0; }
    let q   = cross(s, e1);
    let v   = inv * dot(rd, q);
    if v < 0.0 || (u + v) > 1.0 { return -1.0; }
    let t   = inv * dot(e2, q);
    if t > 1e-4 { return t; }
    return -1.0;
}

// ── Scene ray march ───────────────────────────────────────────────────────────

fn scene_hit(ro: vec3<f32>, rd: vec3<f32>, max_t: f32) -> f32 {
    var closest = max_t;
    for (var mi = 0u; mi < params.num_meshes; mi++) {
        let mesh = meshes[mi];
        for (var ii = mesh.index_offset; ii < mesh.index_offset + mesh.index_count; ii += 3u) {
            let i0 = indices[ii];
            let i1 = indices[ii + 1u];
            let i2 = indices[ii + 2u];
            let p0 = (mesh.transform * vec4<f32>(vertices[i0].pos, 1.0)).xyz;
            let p1 = (mesh.transform * vec4<f32>(vertices[i1].pos, 1.0)).xyz;
            let p2 = (mesh.transform * vec4<f32>(vertices[i2].pos, 1.0)).xyz;
            let t  = ray_triangle(ro, rd, p0, p1, p2);
            if t > 0.0 && t < closest { closest = t; }
        }
    }
    return closest;
}

// ── Light contribution ────────────────────────────────────────────────────────

fn eval_direct(pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    var lo = vec3<f32>(0.0);
    for (var li = 0u; li < params.num_lights; li++) {
        let lgt = lights[li];
        var l_dir = vec3<f32>(0.0);
        var l_dist = params.max_ray_dist;
        var l_radiance = lgt.color_intensity.xyz * lgt.color_intensity.w;

        if lgt.kind == 0u { // directional
            l_dir  = -normalize(lgt.dir_outer.xyz);
            l_dist  = params.max_ray_dist;
        } else { // point / spot
            let to_light = lgt.pos_range.xyz - pos;
            l_dist = length(to_light);
            if l_dist > lgt.pos_range.w { continue; }
            l_dir  = to_light / l_dist;
            let atten = 1.0 / (l_dist * l_dist + 0.0001);
            l_radiance *= atten;
        }

        let ndotl = max(dot(n, l_dir), 0.0);
        if ndotl <= 0.0 { continue; }

        // Shadow ray
        let shadow_t = scene_hit(pos + n * 0.001, l_dir, l_dist - 0.001);
        if shadow_t < l_dist - 0.001 { continue; }

        lo += l_radiance * ndotl;
    }
    return lo;
}

// ── Texel world-space lookup ──────────────────────────────────────────────────

struct TexelInfo { pos: vec3<f32>, normal: vec3<f32>, valid: bool }

fn texel_world_pos(lm_uv: vec2<f32>) -> TexelInfo {
    // Brute-force: find the triangle that contains this LM UV.
    // Production would use a pre-built texel→triangle table.
    var best_t = TexelInfo(vec3<f32>(0.0), vec3<f32>(0.0, 1.0, 0.0), false);
    for (var mi = 0u; mi < params.num_meshes; mi++) {
        let mesh = meshes[mi];
        for (var ii = mesh.index_offset; ii < mesh.index_offset + mesh.index_count; ii += 3u) {
            let i0 = indices[ii]; let i1 = indices[ii+1u]; let i2 = indices[ii+2u];
            let uv0 = vertices[i0].lm_uv;
            let uv1 = vertices[i1].lm_uv;
            let uv2 = vertices[i2].lm_uv;
            // Barycentric coords in UV space
            let d1  = uv1 - uv0; let d2 = uv2 - uv0; let dp = lm_uv - uv0;
            let inv = 1.0 / (d1.x * d2.y - d1.y * d2.x);
            let u   = (dp.x * d2.y - dp.y * d2.x) * inv;
            let v   = (d1.x * dp.y - d1.y * dp.x) * inv;
            if u >= 0.0 && v >= 0.0 && (u + v) <= 1.0 {
                let w = 1.0 - u - v;
                let p0 = (mesh.transform * vec4<f32>(vertices[i0].pos, 1.0)).xyz;
                let p1 = (mesh.transform * vec4<f32>(vertices[i1].pos, 1.0)).xyz;
                let p2 = (mesh.transform * vec4<f32>(vertices[i2].pos, 1.0)).xyz;
                let n0 = (mesh.transform * vec4<f32>(vertices[i0].normal, 0.0)).xyz;
                let n1 = (mesh.transform * vec4<f32>(vertices[i1].normal, 0.0)).xyz;
                let n2 = (mesh.transform * vec4<f32>(vertices[i2].normal, 0.0)).xyz;
                best_t.pos    = p0 * w + p1 * u + p2 * v;
                best_t.normal = normalize(n0 * w + n1 * u + n2 * v);
                best_t.valid  = true;
                return best_t;
            }
        }
    }
    return best_t;
}

// ── Main ──────────────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.resolution;
    if gid.x >= res || gid.y >= res { return; }

    // Seed RNG per-texel
    rng_state = gid.x + gid.y * res + params.frame_seed * res * res;

    let lm_uv = (vec2<f32>(gid.xy) + 0.5) / f32(res);
    let texel  = texel_world_pos(lm_uv);

    if !texel.valid {
        textureStore(out_lm, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    var accum = vec3<f32>(0.0);
    let spp   = params.samples_per_texel;

    for (var s = 0u; s < spp; s++) {
        // Direct illumination
        var radiance = eval_direct(texel.pos, texel.normal);

        // Indirect bounces
        var pos    = texel.pos;
        var normal = texel.normal;
        var throughput = vec3<f32>(1.0);

        for (var b = 0u; b < params.bounce_count; b++) {
            let dir = rand_hemisphere(normal);
            let t   = scene_hit(pos + normal * 0.001, dir, params.max_ray_dist);
            if t >= params.max_ray_dist { break; }

            // Move to bounce point (no material lookup → assume 50% grey diffuse)
            pos    = pos + dir * t;
            normal = dir; // approximate; a real baker would interpolate vertex normals
            throughput *= 0.5 * 3.14159; // Lambertian BRDF × pi
            radiance += throughput * eval_direct(pos, normal);
        }

        accum += radiance;
    }

    if spp > 0u { accum /= f32(spp); }
    textureStore(out_lm, vec2<i32>(gid.xy), vec4<f32>(accum, 1.0));
}
"#;
