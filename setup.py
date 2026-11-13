from setuptools import setup

setup(
    name="online-voting-system",
    version="1.0.0",
    py_modules=["app", "face_utils", "blockchain"],
    entry_points={
        "console_scripts": [
            "Online-voting-system=app:main",
            "online-voting-system=app:main",
        ],
    },
)

