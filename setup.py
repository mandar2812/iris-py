"""Setup for Iris ML.

Author: Mandar Chandorkar
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="iris_py",
    author="Mandar Chandorkar",
    author_email="mandar2812@gmail.com",
    description="A package for training classifiers on the IRIS dataset.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mandar2812/iris-py",
    project_urls={
        "Bug Tracker": "https://github.com/mandar2812/iris-py/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    version_config={
        "template": "{tag}",
        "dev_template": "{tag}.post{ccount}+git.{sha}",
        "dirty_template": "{tag}.post{ccount}+git.{sha}.dirty",
        "starting_version": "0.1.0",
        "version_callback": None,
        "version_file": None,
        "count_commits_from_version_file": False,
    },
    setup_requires=["setuptools-git-versioning"],
    install_requires=["numpy", "scikit-learn", "pandas", "matplotlib"],
    extras_require={
        "dev": [
            "pytest",
            "pytest-mock",
            "types-setuptools",
            "types-requests",
            "types-pytz",
            "types-PyYAML",
            "jupyter",
            "mypy",
            "black",
            "pycodestyle",
            "pydocstyle",
            "pre-commit",
            "nbqa",
            "isort",
        ]
    },
    include_package_data=True,
    package_data={"": ["data/*"]},
)
