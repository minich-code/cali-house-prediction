# Import libraries 
from setuptools import find_packages, setup 
from typing import List 

HYPHEN_E_DOT_REQUIREMENT = '-e .'

# Create a function that will read and return a list of requirements from requirements.txt file  
def get_requirements(file_path:str) -> List[str]:
    # Create an empty list to store the requirements 
    requirements = []

    # Open the requirements.txt file 
    with open(file_path) as file_obj:
        # Iterate over each line in the file 
        for line in file_obj:
            # Remove the newline character at end of each line 
            requirements.append(line.replace("\n", "")) 

    # Remove the '-e .' requirement 
    if HYPHEN_E_DOT_REQUIREMENT in requirements:
        requirements.remove(HYPHEN_E_DOT_REQUIREMENT)

    return requirements
            
setup(
    name="House Price Prediction",
    version="0.1",
    packages=find_packages(),
    author="Western",
    author_email="minichworks@gmail.com",
    url="https://github.com/minich-code/cali-house-prediction/tree/main",
    license="MIT",
    description="A house price prediction model",
    install_requires= get_requirements('requirements.txt')
)