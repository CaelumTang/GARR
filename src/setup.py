from setuptools import find_packages, setup

setup(
    name="garr",
    version="0.1.0",
    description="Generative Alignment and Retrieval Refinement for micro-video popularity prediction",
    author="Xianhe Tang, Jing Yi, Hongchen Wei, Jiayi Xie, Zhenzhong Chen",
    license="Apache-2.0",
    python_requires=">=3.10,<3.11",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "garr-prepare=garr.data.prepare:main",
            "garr-preprocess-microlens=garr.data.microlens:main",
            "garr-preprocess-topicvid=garr.data.topicvid:main",
            "garr-build-vision=garr.data.vision:main",
            "garr-infer-mllm=garr.mllm.infer:main",
            "garr-postprocess-mllm=garr.mllm.postprocess:main",
            "garr-pack-mllm=garr.mllm.pack:main",
            "garr-retrieve=garr.retrieval.retrieve:main",
            "garr-train-predictor=garr.predictor.train:main",
        ]
    },
)
