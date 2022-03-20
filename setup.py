from setuptools import setup

setup(
    name='deep-sudoku',
    version='0.1',
    packages=['deep-sudoku', 'deep-sudoku.utils'],
    url='https://github.com/dtonderski/Deep-Sudoku',
    license='GNU GPLv3',
    author='davton',
    author_email='dtonderski@gmail.com',
    description='Solving Sudokus using a Neural '
                'Network assisted Monte-Carlo approach.',
    install_requires=['numpy', 'py-sudoku']
)
