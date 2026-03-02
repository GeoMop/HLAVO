# DVC usage

[DVC](https://dvc.org) is used to separate version control of large datasets from processing codes and configurations stored on GitHub.
The large development data stored on Google team/shared drive "HLAVO", you need to get read access before using DVC.


## DVC Setup 

1. Use 'dev/dvc_install.sh' for pip install into a Python environment (independent of the project environment) 
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

2. Add google drive remote [HLAVO/Deep Vadose Zone/GIS and large data](https://drive.google.com/drive/u/1/folders/1jx5F59QX6Yc51oFls-Tu_NW1SvmujcQw)

        ```
        dvc remote add -d gdrive gdrive://1jx5F59QX6Yc51oFls-Tu_NW1SvmujcQw

        ```

The hash comes from the link.
