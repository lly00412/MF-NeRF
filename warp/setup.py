from setuptools import setup, Extension

setup(
    name="warptools",
    ext_modules=[
        Extension("warptools", ["render_tgt_volume.cpp"])]
    )