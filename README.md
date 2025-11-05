# Project Template for Deep Learning

## Requirements
*   uv package manager is required.
*   How to install uv :
    - Linux : 
    
        ```sudo apt install uv```
        ```sudo pacman -S install uv```
    
    - Mac :
        
        ```brew install uv```

    - Windows :

       ```powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"```

## Setting up

*   ### Simple setup

    use following command to set up
    and start using uv as a package manager

    ```
    uv venv --python 3.11
    ./.venv/Scripts/activate
    uv lock
    ```


## Managements

*   ### Requirements
  
    use following command to automatically export requirements

    ```
    uv export --format requirements-txt --all-groups --output-file requirements.txt
    ```

*   ### Formatting and Linting

    use following command for auto-formatting using ruff

    ```
    uv run ruff format .
    ```

    use following command for checking rules

    ```
    uv run ruff check .
    ```

    use *--fix* option for auto-fix in check.

*   ### Using precommits

    use following command for checking rules and exporting all the requirements

    ```
    uv run pre-commit run --all-files -v
    ```