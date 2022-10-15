from setuptools import setup

setup(
    name='deepsudoku',
    version='0.3.9',

    packages=['deepsudoku', 'deepsudoku.utils'],
    url='https://github.com/dtonderski/DeepSudoku',
    license='GNU GPLv3',
    author='davton',
    author_email='dtonderski@gmail.com',
    description='Solving Sudokus using a Neural '
                'Network assisted Monte-Carlo approach.',
    install_requires=['numpy', 'py-sudoku']
)
