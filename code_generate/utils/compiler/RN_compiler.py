import subprocess
import os
import threading
import time
import re
import json
import shutil

   
def strip_ansi_escape(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def is_package_installed(package_name, package_json_path):
    # make sure package.json exists
    if not os.path.exists(package_json_path):
        print(f"Error: {package_json_path} does not exist.")
        return False

    try:
        with open(package_json_path, 'r') as file:
            package_data = json.load(file)
    except json.JSONDecodeError:
        print("Error: package.json is not a valid JSON file.")
        return False

    # inspect dependencies and devDependencies
    dependencies = package_data.get('dependencies', {})
    dev_dependencies = package_data.get('devDependencies', {})

    # Check if the package is listed in any dependency list.
    is_installed = package_name in dependencies or package_name in dev_dependencies
    return is_installed
   
def run_gradlew_assemble(project_path,category,apk_name,add_package_list=[]):
    original_cwd = os.getcwd()
    error_dict = ""
    compile_state = False
    if not os.path.isdir(project_path):
        error_dict = f"Error: React Native project path does not exist or is not a directory: {project_path}"
        print(error_dict)
        return False, error_dict

    os.chdir(project_path)
    if not os.path.exists('android'):
        error_dict = (
            "Error: The current directory is not the root of a React Native project. "
            f"project_path={project_path}"
        )
        print(error_dict)
        os.chdir(original_cwd)
        return False, error_dict

    os.chdir('android')

    # inspect whether gradlew exists
    if not os.path.exists('gradlew'):
        error_dict = f"Error: gradlew file not exists in {os.getcwd()}."
        print(error_dict)
        os.chdir(original_cwd)
        return False, error_dict
        print("Error：gradlew file not exists.")
        return
    npm_install_cmd = [r"C:\nvm4w\nodejs\npm.cmd", "install"]
    npm_uninstall_cmd  = [r"C:\nvm4w\nodejs\npm.cmd", "uninstall"]
    package_json_path = os.path.join(project_path, "package.json")
    # If there are new packages, install them using npm install.
    if add_package_list:
        for add_package in add_package_list:
            if not is_package_installed(add_package, package_json_path):
                npm_install_cmd.append(add_package)
                npm_uninstall_cmd.append(add_package)
        print(npm_install_cmd)
        if len(npm_install_cmd) > 2:
            subprocess.run(npm_install_cmd, check=True)
    try:
        # run gradlew assemble command
        result = subprocess.run(['gradlew.bat', 'assemble'], 
                                capture_output=True, 
                                text=True, 
                                check=True)
        print("gradlew assemble run successfully")
        compile_state = True
    except subprocess.CalledProcessError as e:
        # If the command execution fails, write the error information to err_log.txt.
        print(f"gradlew assemble failed to run。Error msg：{e.stderr}")
        compile_state = False
        clean_error = strip_ansi_escape(e.stderr)
        parsed_error = extract_error_info(clean_error)
        if isinstance(parsed_error, dict) and parsed_error:
            error_dict = parsed_error
            for key, value in error_dict.items():
                print(f"file: {key}\n Error: {value}\n")
        else:
            error_dict = clean_error
        with open('../err_log.txt', 'w', encoding='utf-8') as f:
            f.write(clean_error)
        print(f"The gradlew assemble command failed. The error log has been saved to err_log.txt.")  
    # If there are new packages, they need to be removed after the build is complete to avoid affecting the next compilation.
    if add_package_list and len(npm_uninstall_cmd)>2:
        subprocess.run(npm_uninstall_cmd, check=True)   
    os.chdir(original_cwd)
    return compile_state,error_dict
 
def extract_error_info(log_text):
    pattern = r'(\w+\.js): (.+?)\n\n\s*((?:\d+\s*\|\s*.*(?:\n|$))+)'

    matches = re.finditer(pattern, log_text, re.MULTILINE | re.DOTALL)
    
    error_dict = {}
    for match in matches:
        filename = match.group(1)
        error_description = match.group(2).strip()
        error_info = match.group(3)
        
        error_lines = error_info.split('\n')
        cleaned_error_info = "\n".join(line.strip() for line in error_lines if line.strip() and '|' in line)

        full_error_info = f"{error_description}\n{cleaned_error_info}"

        if filename in error_dict:
            error_dict[filename] += "\n\n" + full_error_info
        else:
            error_dict[filename] = full_error_info
    
    return error_dict

 

def copy_file2project(src_folder, dst_folder):
    components_folder = os.path.join(dst_folder,"components")
    for item in os.listdir(src_folder):
        extension = os.path.splitext(item)[1]
        if extension == '.js' or extension == '.jsx':
            if item == "App.js":
                shutil.copy(os.path.join(src_folder, item), dst_folder)
            else:
                src_path = os.path.join(src_folder, item)
                shutil.copy(src_path, components_folder)
                print(f"{item} copied!")   
                
if __name__ == "__main__":
    app_classification_list = ["AI"]
    app_classification = app_classification_list[0]
    app_name = 'ChatGPT' 
    src_folder = r""   
    project_path = r"" 
    copy_file2project(src_folder,project_path)
    run_gradlew_assemble(project_path,app_classification,app_name,add_package_list=None)
