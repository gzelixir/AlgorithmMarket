

import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--data', type=str, help='csv path', default = "mechanical.csv")
args = parser.parse_args()


if __name__ == '__main__':
    def check_and_install_package(package_name):
        # Check if the package is available in the current environment using Conda
        conda_check_cmd = f"conda list {package_name}"
        result = subprocess.run(conda_check_cmd, shell=True, capture_output=True, text=True)
        print("result.stdout:", result.stdout)
        
        if package_name not in result.stdout:
            print("installing physo")
            # If the package is not available, install it using pip
            pip_install_cmd = f"pip install -e ."
            os.system(pip_install_cmd)
        else:
            print("physo have been installed")

    # Replace 'physo' with the actual package name you want to check and install
    check_and_install_package("physo")
    
    
    
    # texlive_path = os.path.join(__file__.split("demo")[0],"latex")
    texlive_path = "/user/mahaohui/autoML/PhySO/latex"
    print("texlive_path:",texlive_path)
    os.system(f"export MANPATH={texlive_path}/texlive/2023/texmf-dist/doc/man:$MANPATH")
    os.system(f"export INFOPATH={texlive_path}/texlive/2023/texmf-dist/doc/info:$INFOPATH")
    os.system(f"export PATH={texlive_path}/texlive/2023/bin/x86_64-linux:$PATH")
    

    os.system(f'python demo_quick_sr.py --data {args.data}')
