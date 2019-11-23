Study AI with Python
===

## Setup Python environment

### Python version '3.7.4'

```bash
$ pyenv local 3.7.4
```

### Make virtualenv

```bash
$ python -m venv .venv --prompt='ai'
$ source .venv/bin/activate
```

### Install required packages

```bash
$ pip install -r requirements.txt
```

## Setup Jupiter notebook

### Install code formatter

```bash
$ jupyter labextension install @ryantam626/jupyterlab_code_formatter
$ jupyter serverextension enable --py jupyterlab_code_formatter
```

### Setup autopep8 code formatter

1. Startup jupyter lab

   ```bash
   $ jupyter lab
   ```

2. In 'Web browser', open menu 'Settings -> Advanced Settings Editor'

3. In 'Jupyterlab Code Formatter' section, add the following configs

   ```json
   {
       "autopep8": {
           "max_line_length": 120,
           "ignore": [
               "E226",
               "E302",
               "E41"
           ]
       }
   }
   ```

4. In 'Keyboarders shortcuts' section, add the following config

   ```json
   {
       "shortcuts": [
           {
               "command": "jupyterlab_code_formatter:autopep8",
               "keys": [
                   "Ctrl K",
                   "Ctrl M"
               ],
               "selector": ".jp-Notebook.jp-mod-editMode"
           }
       ]
   }
   ```

   