use burn::backend::{Autodiff, Wgpu};
use burn::backend::wgpu::AutoGraphicsApi;
use burn::optim::AdamConfig;
use crate::this_is_burn::model::{ModelConfig};


mod this_is_burn;



fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    this_is_burn::training::train::<MyAutodiffBackend>(
        "E:\\RustPrograms\\burn-rs\\my-first-burn-model\\tmp\\guide",
        this_is_burn::training::TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device,
    );
}