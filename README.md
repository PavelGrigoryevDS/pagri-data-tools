<!-- Убрать лишние пакеты из toml -->

# Description  
This repository contains data analysis modules, including tools for data preprocessing, visualization, statistical analysis. The modules can be used to tackle a wide range of data analysis tasks.

***
 
### Quick Steps to Use Module with Poetry
#### 1. Install Poetry:  
This command installs Poetry on your system.
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

#### 3. Create a New Project:  
This command creates a new directory named my_new_project with a basic project structure.
```bash
poetry new my_new_project
```

#### 4. Navigate to the Project Directory:  
Change into the newly created project directory.
```bash
cd my_new_project
```

#### 5. Add Module as a Dependency:  
This command adds module from GitHub as a dependency to the project.
```bash
poetry add git+https://github.com/PAGriAnalytics/pagri-data-tools.git
```

#### 6. Install Dependencies:  
This command installs all dependencies listed in the pyproject.toml file.
```bash
poetry install
```

#### 6. Update Dependencies:
This command updates all dependencies listed in the pyproject.toml file to their latest compatible versions, respecting the version constraints.
```bash
poetry update
```

#### 8. Import Module:  
You can now import and use module in your Python scripts.
```python
import pagri_data_tools as pgdt
```





