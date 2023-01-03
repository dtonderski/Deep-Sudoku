from setuptools import setup, find_packages

setup(
    name='deepsudoku',
    version='0.9.0',

    packages=find_packages(),
    url='https://github.com/dtonderski/DeepSudoku',
    license='GNU GPLv3',
    author='davton',
    author_email='dtonderski@gmail.com',
    description='Solving Sudokus using a Neural '
                'Network assisted Monte-Carlo approach.',
    install_requires=['numpy', 'py-sudoku', 'einops', 'torch']
)
