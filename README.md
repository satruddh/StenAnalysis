# Running or Installing
- You can use the docker to completely run in an isolated fashion.
- Clone this repo.
- Navigate to the cloned repo directory 
- Open terminal in this directory and issue the following commands
    ```
    docker compose build
    docker compose up
    ```
- ALternatively, if just want to test, you can issue the following command: 
    `python app.py`

When running for the first time, open `http://localhost:5000/login` and register a doctor account. Use this account to log in and access the analysis dashboard.
