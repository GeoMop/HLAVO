# DVC usage

[DVC](https://dvc.org) is used to separate version control of large datasets from processing codes and configurations stored on GitHub.
For endorse repository the large datasets are stored on Google drive under the shared drive DZ04_Chodby.

## DVC Setup 

1. Use 'bin/dvc_install.sh' for pip install into a Python environment 
   or [DVC install](https://dvc.org/doc/install) for other options like install it system-wide.
   
2. Download DVC endorse secret [config script](https://drive.google.com/file/d/1Dag4N3KYz5q9rkLURayXHjUV0yN-zYYH/view?usp=drive_link),
   place it to endorse root under original name (NEVER COMMIT THIS FILE).

3. Execute the script like:

        ```
        bash dvc_secret_config.sh
        ```

4. Pull the large files:
        ```
        dvc pull
        ```
   The browser should open to ask you for the login to your Google account (the TUL one ussually).
   
   
See [large datasets modification doc](https://dvc.org/doc/user-guide/data-management/modifying-large-datasets) for further work.

## Adding remote (initialization)

1. Initialize `.dvc` folder. From the repository root run:

        ```
        dvc init
        ``` 

2. Add google drive remote [DZ04_Chodby/Podklady/endorse_large_files](https://drive.google.com/drive/u/1/folders/109cr1pZ8GV5s8yXKgVzl8NPQ8j537E4T)

        ```
        dvc remote add -d gdrive gdrive://109cr1pZ8GV5s8yXKgVzl8NPQ8j537E4T

        ```

The hash comes form the link.
