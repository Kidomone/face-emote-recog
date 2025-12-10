import os

create_scripts = [
    "create_db_users.py",
    "create_db_humans.py",
    "create_db_user_photos.py",
    "create_db_uploaded_images.py",
    "create_db_face_detection.py",
]


def run_script(script_name):
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(f"Выполняем {script_name}...")
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            code = compile(f.read(), script_name, "exec")
            exec(code, {"__name__": "__main__"})
        print(f"{script_name} выполнен успешно.\n")
    except Exception as e:
        print(f"Ошибка при выполнении {script_name}: {e}\n")


def main():
    print("Инициализация базы данных...")
    for script in create_scripts:
        run_script(script)
    print("Все таблицы созданы.")


if __name__ == "__main__":
    main()
