### Bootstrap process
#### Through Github

Go to the recipe project in github and select:
- Use this template > Create a new repository

#### Through code
1. Create a new repo in github named < REPO >
2. clone recipe project to a directory named < REPO >
3.  ```commandline 
    git clone git@github.com:aiChatbotPOC/recipe-project.git <REPO>
4. Change the origins of the repo and clear template history
```commandline 
   make change-origin NEW_URL='<new repo directory>'
  ```

### Create the environment and install dependencies
```commandline
make environment
```

### Update the following files with the project information
1. recipe_project/about.py, tests/test_about.py
2. describe the structure of the project in recipe_project/__init__.py
3. Update main and recipe_project/brain.py file


### Build the project
```commandline
make build
```

### Install the project as a dependency
```commandline
pipenv install /path/to/wheel/distribution
```

### Execute the project from cli [view the __ENTRY_POINTS__ in setup.py]
```commandline
recipe-package-project
```

### Import the project in python
```commandline
import recipe_package_project
```