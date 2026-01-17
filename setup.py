'''
The setup.py file is an essential part of packeging and distributing Python projects. It is used by setuptools (or distutils in older Python versions) to define the configuration of your project, such as its metadata, dependencies, and more.
'''

from setuptools import find_packages,setup
from typing import List

def get_requirements() -> List[str]:
    '''
    This function return list of requirements
    '''
    requirement_lst:List[str] = []
    try:
        with open('requirements.txt', 'r') as file:
            # Read lines from the file
            lines = file.readlines()
            # Process each line
            for line in lines:
                # Remove space from line
                requirement = line.strip()
                # Ignore empty lines and -e.
                if requirement and requirement != '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")
        
    return requirement_lst


setup(
    name = "Network Security System",
    version = "0.0.1",
    author = "Kirtan Patel",
    author_email = "kpmandaviya001@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements(),

)