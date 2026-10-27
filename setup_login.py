import os
import subprocess
import sys

def main():
    profile_dir = os.path.abspath("playwright_profile")

    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe")
    ]

    chrome_exe = None
    for path in chrome_paths:
        if os.path.exists(path):
            chrome_exe = path
            break

    if not chrome_exe:
        print("Google Chrome не найден по стандартным путями. Пожалуйста, убедитесь, что он установлен.")
        sys.exit(1)

    print(f"Найден Chrome: {chrome_exe}")
    print(f"Создаем/открываем изолированный профиль для бота в: {profile_dir}")
    print("=" * 60)
    print("ИНСТРУКЦИЯ ПО АВТОРИЗАЦИИ:")
    print("1. Сейчас откроется обычное окно Google Chrome (без автоматизации).")
    print("2. Войдите в свой Google аккаунт.")
    print("3. Перейдите на https://gemini.google.com и убедитесь, что чат загружается.")
    print("4. ПОЛНОСТЬЮ ЗАКРОЙТЕ это окно браузера.")
    print("5. После этого сессия сохранится, и бот (или test_playwright.py) сможет работать в фоне!")
    print("=" * 60)

    try:
        # Запускаем реальный Chrome с папкой профиля, которую использует Playwright
        subprocess.run([chrome_exe, f"--user-data-dir={profile_dir}", "https://gemini.google.com/app"])
    except Exception as e:
        print(f"Ошибка при запуске Chrome: {e}")

if __name__ == "__main__":
    main()
