from setuptools import find_packages, setup

setup(name='v2realbot_research',
      version='0.1',
      description='Research for v2realbot',
      author='David Brazda',
      author_email='davidbrazda61@gmail.com',
      packages=find_packages(),
      install_requires=[
            'pandas',
            'pywebview>=5.0.5',
            'orjson',
            'v2trading @ git+https://github.com/drew2323/v2trading.git@master#egg=v2trading',
            'lightweight-charts-python @ https://github.com/drew2323/lightweight-charts-python.git@main#egg=lightweight-charts-python'
      ]
     )