from database_service import create_or_migrate_database
from config import CONFIG


def main():
    create_or_migrate_database()
    print("Da tao/cap nhat database thanh cong:", CONFIG.database_file)


if __name__ == "__main__":
    main()
