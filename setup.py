from setuptools import setup, find_packages
setup(
        name='flows_diagrams',
        python_requires=">=3.10",
        version='0.0',
        packages=find_packages(),
        install_requires=[
   'jaxlib==0.4.23',
   'jax==0.4.23',
   'jax-md==0.2.8',
   'equinox==0.11.4',
   'matplotlib==3.7.0',
   'numpy==1.25.2',
   'mdtraj==1.10.0',
   'scipy==1.9.3',
]
)
