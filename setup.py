from setuptools import find_namespace_packages, setup


setup(
    name="lcp-pinn",
    version="0.1.0",
    description="Action-conditioned Contact World Model for Nero robot dynamics",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10,<3.12",
    packages=find_namespace_packages(
        include=[
            "data_process*",
            "model*",
            "physics*",
            "scripts*",
            "train*",
        ]
    ),
    install_requires=[
        # Pinocchio 3.9 / cmeel-boost on Python 3.10 requires NumPy 2.2.
        "numpy>=2.2,<2.3",
        "scipy>=1.14,<2",
        "PyYAML>=6.0,<7",
        "tqdm>=4.65,<5",
        "matplotlib>=3.7,<4",
        "h5py>=3.9,<4",
        # Zarr v2 keeps the synchronous local-store API used by the training
        # workers; Zarr v3's async bridge can block during group creation.
        "zarr>=2.16,<3",
        "pandas>=2.0,<3",
        "pyarrow>=14,<20",
        "lerobot==0.4.0",
        "huggingface-hub[cli,hf-transfer]>=0.34.2,<0.36.0",
        "fsspec[http]>=2023.1.0,<=2025.3.0",
        "packaging>=24.2,<26.0",
        "wandb>=0.19,<0.27",
    ],
    extras_require={
        "physics": [
            "cmeel-tinyxml2==10.0.0",
            "cmeel-urdfdom==4.0.1",
            "libpinocchio==3.9.0",
            "pin==3.9.0",
            "mujoco>=3.1.3,<3.4.0",
        ],
        "vision": [
            "opencv-python>=4.8,<5",
            "open3d>=0.18,<1",
            "Pillow>=10,<12",
            "pyrealsense2",
            "ultralytics",
        ],
        "test": [
            "pytest>=7.4,<9",
        ],
    },
    entry_points={
        "console_scripts": [
            "pinn-build-offline-labels=data_process.tool.build_offline_tau_labels:main",
            "pinn-train-tau-other=train.trainer.tau_other_sequence_train:main",
            "pinn-train-tau-free-v2=train.trainer.tau_free_sequence_train_v2:main",
            "pinn-train-contact-wm=train.trainer.contact_world_model_train:main",
            "pinn-train-contact-wm-opd=train.trainer.contact_world_model_opd_train:main",
            "pinn-contact-wm-rollout=data_process.tool.contact_world_model_rollout_visualizer:main",
        ]
    },
)
