from setuptools import setup, find_packages

setup(
    name='mimo-dataset-generator',
    version='0.1.0',
    description='Advanced MIMO Channel Dataset Generation Framework',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Milad',  
    author_email='snatanzi@wpi.edu',  
    url='https://github.com/yourusername/mimo-dataset-generator',
    packages=find_packages(),
    install_requires=[
        # Core Scientific Computing and Deep Learning
        'numpy>=1.22.0,<2.0.0',
        'tensorflow>=2.13.0,<2.16.0',  # Align with Sionna recommendations
        'tensorflow-probability>=0.20.0,<0.21.0',  # Compatible with TensorFlow 2.13+
        
        # Wireless Communication Simulation
        'sionna>=0.15.0,<0.16.0',  # Ensure compatibility with TensorFlow
        
        # Data Handling and Visualization
        'h5py>=3.7.0,<4.0.0',
        'scipy>=1.9.0,<2.0.0',
        'matplotlib>=3.6.0,<4.0.0',
        
        # Progress Tracking and Logging
        'tqdm>=4.64.0,<5.0.0',
        
        # Optional: Machine Learning and Statistical Analysis
        'scikit-learn>=1.1.0,<2.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.1.0,<8.0.0',
            'flake8>=4.0.0,<5.0.0',
            'black>=22.3.0,<23.0.0',
            'mypy>=0.971,<1.0.0',
            'sphinx>=5.0.0,<6.0.0',  # Documentation tools
        ],
        'gpu': [
            'tensorflow-gpu>=2.13.0,<2.16.0',  # Ensure GPU version compatibility
            'cuda-toolkit>=11.8,<12.0',
            'cudnn>=8.6,<9.0',
        ],
    },
    python_requires='>=3.8,<3.12',  # Updated for Sionna and TensorFlow compatibility
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Communications :: Telephony',
    ],
    keywords='mimo channel dataset generation wireless communication 5g 6g machine learning tensorflow sionna',
    entry_points={
        'console_scripts': [
            'mimo-dataset-gen=main:main',
        ],
    },
    package_data={
        '': ['README.md', 'LICENSE'],
    },
    include_package_data=True,
)
