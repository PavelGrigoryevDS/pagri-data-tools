from setuptools import setup

setup(
    name='pagri_data_tools',  # Имя вашего пакета
    version='0.1',  # Версия вашего пакета
    install_requires=[
        'dash'
        'pingouin'
        'pyaspeller'
        'pymystem3'
        ],
    description='This repository contains data analysis modules, including tools for data preprocessing, visualization, statistical analysis.',  # Описание
    long_description=open('README.md').read(),  # Долгое описание из README
    long_description_content_type='text/markdown',  # Тип контента
    author='Pavel Grigoryev',  # Ваше имя
    author_email='pagri.analytics@gmail.com',  # Ваш email
    url='https://github.com/PAGriAnalytics/pagri_analytics_modules',  # URL репозитория
)
