use async_trait::async_trait;
use nebula_core::{
    context::BakeContext, error::NebulaError, progress::ProgressReporter,
    scene::SceneGeometry, traits::BakePass,
};
use nebula_gpu::texture::{BakeTexture, TextureFormat2D};
use crate::{config::AoConfig, output::AoOutput};

#[derive(Default)]
pub struct AoBaker;

#[async_trait]
impl BakePass for AoBaker {
    type Input  = AoConfig;
    type Output = AoOutput;

    fn name(&self) -> &'static str { "ao" }

    async fn execute(
        &self,
        scene:    &SceneGeometry,
        config:   &AoConfig,
        ctx:      &BakeContext,
        reporter: &dyn ProgressReporter,
    ) -> Result<AoOutput, NebulaError> {
        let res = config.resolution.clamp(64, nebula_gpu::MAX_TEXTURE_DIM);
        reporter.begin("ao", 3);

        reporter.step("ao", 0, "uploading scene geometry");
        let (vbuf, ibuf, mbuf) = upload_geometry(scene, ctx)?;

        reporter.step("ao", 1, &format!("baking {}×{} AO ({} rays)", res, res, config.ray_count));
        let ao_tex = BakeTexture::new(
            &ctx.device, "nebula_ao", res, res, TextureFormat2D::R32F, 1,
            wgpu::TextureUsages::empty(),
        );
        dispatch_ao(scene, config, ctx, &ao_tex, &vbuf, &ibuf, &mbuf)?;

        reporter.step("ao", 2, "reading back");
        let texels = ao_tex.read_back(&ctx.device, &ctx.queue);
        let config_json = serde_json::to_string(config).unwrap_or_default();

        reporter.finish("ao", true, "done");
        Ok(AoOutput { width: res, height: res, texels, config_json })
    }
}

fn upload_geometry(
    scene: &SceneGeometry,
    ctx:   &BakeContext,
) -> Result<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer), NebulaError> {
    use wgpu::util::DeviceExt;
    use bytemuck::{Pod, Zeroable};

    #[repr(C)] #[derive(Copy,Clone,Pod,Zeroable)]
    struct GpuVert { pos: [f32;3], _p: f32, normal: [f32;3], _p2: f32, lm_uv: [f32;2], _p3: [f32;2] }

    #[repr(C)] #[derive(Copy,Clone,Pod,Zeroable)]
    struct GpuMesh { idx_off: u32, idx_cnt: u32, vert_off: u32, _p: u32, xform: [[f32;4];4] }

    let mut verts: Vec<GpuVert> = Vec::new();
    let mut idxs:  Vec<u32>     = Vec::new();
    let mut meshes: Vec<GpuMesh> = Vec::new();

    for mesh in &scene.meshes {
        let vb = verts.len() as u32;
        let ib = idxs.len()  as u32;
        let fallback: Vec<[f32;2]> = mesh.uvs.clone();
        let lm = mesh.lightmap_uvs.as_ref().unwrap_or(&fallback);
        for i in 0..mesh.positions.len() {
            verts.push(GpuVert {
                pos:    mesh.positions[i], _p: 0.0,
                normal: mesh.normals.get(i).copied().unwrap_or([0.0,1.0,0.0]), _p2: 0.0,
                lm_uv:  lm.get(i).copied().unwrap_or([0.0;2]), _p3: [0.0;2],
            });
        }
        idxs.extend(mesh.indices.iter().map(|&i| i + vb));
        meshes.push(GpuMesh { idx_off: ib, idx_cnt: mesh.indices.len() as u32, vert_off: vb, _p: 0, xform: mesh.world_transform.0.to_cols_array_2d() });
    }

    let mk = |label, data: &[u8]| ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage: wgpu::BufferUsages::STORAGE });
    Ok((
        mk("nebula_ao_vbuf",  bytemuck::cast_slice(&verts)),
        mk("nebula_ao_ibuf",  bytemuck::cast_slice(&idxs)),
        mk("nebula_ao_mbuf",  bytemuck::cast_slice(&meshes)),
    ))
}

fn dispatch_ao(
    scene:   &SceneGeometry,
    config:  &AoConfig,
    ctx:     &BakeContext,
    ao_tex:  &BakeTexture,
    vbuf:    &wgpu::Buffer,
    ibuf:    &wgpu::Buffer,
    mbuf:    &wgpu::Buffer,
) -> Result<(), NebulaError> {
    use bytemuck::{Pod, Zeroable};
    use wgpu::util::DeviceExt;

    #[repr(C)] #[derive(Copy,Clone,Pod,Zeroable)]
    struct AoParams { res: u32, ray_count: u32, max_dist: f32, bias: f32, num_meshes: u32, num_indices: u32, seed: u32, _p: u32 }

    let total_idx: usize = scene.meshes.iter().map(|m| m.indices.len()).sum();
    let p = AoParams { res: config.resolution, ray_count: config.ray_count, max_dist: config.max_distance, bias: config.bias, num_meshes: scene.meshes.len() as u32, num_indices: total_idx as u32, seed: 0xDEADBEEF, _p: 0 };

    let pbuf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("nebula_ao_params"), contents: bytemuck::bytes_of(&p), usage: wgpu::BufferUsages::UNIFORM });

    let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("nebula_ao_bgl"),
        entries: &[
            bgl_uniform(0), bgl_storage_ro(1), bgl_storage_ro(2), bgl_storage_ro(3),
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: ao_tex.format, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
        ],
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nebula_ao_bg"), layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: vbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: ibuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: mbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&ao_tex.view) },
        ],
    });

    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("nebula_ao_cs"), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(AO_WGSL)) });
    let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("nebula_ao_pipeline"), layout: Some(&pl), module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });

    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("nebula_ao_enc") });
    { let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("nebula_ao_pass"), timestamp_writes: None });
      pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bg, &[]);
      let wg = nebula_gpu::WORKGROUP_SIZE; let r = config.resolution;
      pass.dispatch_workgroups(r.div_ceil(wg), r.div_ceil(wg), 1); }
    ctx.queue.submit(std::iter::once(enc.finish()));
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    Ok(())
}

fn bgl_uniform(b: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }
}
fn bgl_storage_ro(b: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None }
}

const AO_WGSL: &str = r#"
struct Params { res: u32, ray_count: u32, max_dist: f32, bias: f32, num_meshes: u32, num_indices: u32, seed: u32, _p: u32 }
struct Vertex { pos: vec3<f32>, _p: f32, normal: vec3<f32>, _p2: f32, lm_uv: vec2<f32>, _p3: vec2<f32> }
struct Mesh   { idx_off: u32, idx_cnt: u32, vert_off: u32, _p: u32, xform: mat4x4<f32> }

@group(0) @binding(0) var<uniform>        params:  Params;
@group(0) @binding(1) var<storage, read>  verts:   array<Vertex>;
@group(0) @binding(2) var<storage, read>  indices: array<u32>;
@group(0) @binding(3) var<storage, read>  meshes:  array<Mesh>;
@group(0) @binding(4) var                 out_ao:  texture_storage_2d<r32float, write>;

var<private> rng: u32;
fn pcg() -> u32 { rng = rng*747796405u+2891336453u; let w=(((rng>>(rng>>28u+4u))^rng)*277803737u); return (w>>22u)^w; }
fn rf() -> f32 { return f32(pcg())*(1.0/4294967296.0); }

fn hemisphere(n: vec3<f32>) -> vec3<f32> {
    let u1=rf(); let u2=rf();
    let r=sqrt(max(0.0,1.0-u1*u1)); let phi=6.28318530718*u2;
    var t=vec3<f32>(1.0,0.0,0.0); if abs(n.x)>0.9{t=vec3<f32>(0.0,1.0,0.0);}
    let b=normalize(cross(n,t)); let tt=cross(b,n);
    return normalize(r*cos(phi)*tt+r*sin(phi)*b+u1*n);
}

fn ray_tri(ro:vec3<f32>,rd:vec3<f32>,v0:vec3<f32>,v1:vec3<f32>,v2:vec3<f32>)->f32{
    let e1=v1-v0; let e2=v2-v0; let h=cross(rd,e2); let det=dot(e1,h);
    if abs(det)<1e-7{return -1.0;} let inv=1.0/det; let s=ro-v0;
    let u=inv*dot(s,h); if u<0.0||u>1.0{return -1.0;}
    let q=cross(s,e1); let v=inv*dot(rd,q); if v<0.0||(u+v)>1.0{return -1.0;}
    let t=inv*dot(e2,q); if t>1e-4{return t;} return -1.0;
}

fn scene_hit(ro:vec3<f32>,rd:vec3<f32>,max_t:f32)->bool{
    for(var mi=0u;mi<params.num_meshes;mi++){
        let m=meshes[mi];
        for(var ii=m.idx_off;ii<m.idx_off+m.idx_cnt;ii+=3u){
            let i0=indices[ii];let i1=indices[ii+1u];let i2=indices[ii+2u];
            let p0=(m.xform*vec4<f32>(verts[i0].pos,1.0)).xyz;
            let p1=(m.xform*vec4<f32>(verts[i1].pos,1.0)).xyz;
            let p2=(m.xform*vec4<f32>(verts[i2].pos,1.0)).xyz;
            if ray_tri(ro,rd,p0,p1,p2) < max_t { return true; }
        }
    }
    return false;
}

fn texel_info(lm_uv:vec2<f32>)->vec4<f32>{// xyz=pos w=valid(1=yes)
    for(var mi=0u;mi<params.num_meshes;mi++){
        let m=meshes[mi];
        for(var ii=m.idx_off;ii<m.idx_off+m.idx_cnt;ii+=3u){
            let i0=indices[ii];let i1=indices[ii+1u];let i2=indices[ii+2u];
            let uv0=verts[i0].lm_uv; let uv1=verts[i1].lm_uv; let uv2=verts[i2].lm_uv;
            let d1=uv1-uv0; let d2=uv2-uv0; let dp=lm_uv-uv0;
            let inv=1.0/(d1.x*d2.y-d1.y*d2.x);
            let u=(dp.x*d2.y-dp.y*d2.x)*inv; let v=(d1.x*dp.y-d1.y*dp.x)*inv;
            if u>=0.0&&v>=0.0&&(u+v)<=1.0{
                let w=1.0-u-v;
                let p0=(m.xform*vec4<f32>(verts[i0].pos,1.0)).xyz;
                let p1=(m.xform*vec4<f32>(verts[i1].pos,1.0)).xyz;
                let p2=(m.xform*vec4<f32>(verts[i2].pos,1.0)).xyz;
                return vec4<f32>(p0*w+p1*u+p2*v,1.0);
            }
        }
    }
    return vec4<f32>(0.0,0.0,0.0,0.0);
}

fn texel_normal(lm_uv:vec2<f32>)->vec3<f32>{
    for(var mi=0u;mi<params.num_meshes;mi++){
        let m=meshes[mi];
        for(var ii=m.idx_off;ii<m.idx_off+m.idx_cnt;ii+=3u){
            let i0=indices[ii];let i1=indices[ii+1u];let i2=indices[ii+2u];
            let uv0=verts[i0].lm_uv;let uv1=verts[i1].lm_uv;let uv2=verts[i2].lm_uv;
            let d1=uv1-uv0;let d2=uv2-uv0;let dp=lm_uv-uv0;
            let inv=1.0/(d1.x*d2.y-d1.y*d2.x);
            let u=(dp.x*d2.y-dp.y*d2.x)*inv;let v=(d1.x*dp.y-d1.y*dp.x)*inv;
            if u>=0.0&&v>=0.0&&(u+v)<=1.0{
                let w=1.0-u-v;
                let n0=(m.xform*vec4<f32>(verts[i0].normal,0.0)).xyz;
                let n1=(m.xform*vec4<f32>(verts[i1].normal,0.0)).xyz;
                let n2=(m.xform*vec4<f32>(verts[i2].normal,0.0)).xyz;
                return normalize(n0*w+n1*u+n2*v);
            }
        }
    }
    return vec3<f32>(0.0,1.0,0.0);
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
    let res=params.res;
    if gid.x>=res||gid.y>=res{return;}
    rng = gid.x + gid.y*res + params.seed*res*res;
    let lm_uv=(vec2<f32>(gid.xy)+0.5)/f32(res);
    let ti=texel_info(lm_uv);
    if ti.w<0.5{textureStore(out_ao,vec2<i32>(gid.xy),vec4<f32>(0.0));return;}
    let pos=ti.xyz; let n=texel_normal(lm_uv);
    var unoccluded=0u;
    for(var s=0u;s<params.ray_count;s++){
        let dir=hemisphere(n);
        if !scene_hit(pos+n*params.bias,dir,params.max_dist){unoccluded+=1u;}
    }
    let ao=f32(unoccluded)/f32(params.ray_count);
    textureStore(out_ao,vec2<i32>(gid.xy),vec4<f32>(ao,0.0,0.0,0.0));
}
"#;
