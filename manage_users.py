"""
manage_users.py
================
Terminal tool for creating and managing user accounts.

This is how you create the FIRST admin account - you can't log into
the app to create a user before any user exists yet. Run this once
to bootstrap an admin, then app.py's own "Manage users" panel
(visible only when logged in as an admin) handles everything after
that, including creating regular employee accounts.

Deliberately does NOT import rag_core.py - rag_core.py requires a
working GEMINI_API_KEY just to be imported (it builds an API client
at the top of the file), and creating a user account has nothing to
do with that. This script only needs to know what folders exist on
disk, which it checks directly.

Run:
    python manage_users.py
"""

import os
import getpass
import auth_store

DOCS_FOLDER = "documents"


def list_available_folders():
    if not os.path.isdir(DOCS_FOLDER):
        return []
    return sorted(
        name for name in os.listdir(DOCS_FOLDER)
        if os.path.isdir(os.path.join(DOCS_FOLDER, name))
    )


def create_user_flow():
    username = input("New username: ").strip()
    if not username:
        print("Username can't be empty.")
        return

    # getpass hides the password as it's typed, instead of echoing it
    # to the terminal in plain view - same reason a browser password
    # field shows dots instead of letters.
    password = getpass.getpass("New password (hidden while typing): ")
    if not password:
        print("Password can't be empty.")
        return

    is_admin = input("Make this an admin (sees every department)? (y/n): ").strip().lower() == "y"

    if is_admin:
        folders = "ALL"
    else:
        available = list_available_folders()
        if not available:
            print(f"No folders found under {DOCS_FOLDER}/ yet - creating this user with no access.")
            folders = []
        else:
            print("Available folders:", ", ".join(available))
            raw = input("Which folders should this user access? (comma-separated): ").strip()
            chosen = [f.strip() for f in raw.split(",") if f.strip()]
            unknown = [f for f in chosen if f not in available]
            if unknown:
                print(f"Warning: these don't match an existing folder yet: {', '.join(unknown)}")
                print("(Still saved - useful if you're about to add that folder.)")
            folders = chosen

    try:
        auth_store.create_user(username, password, folders, db_file=auth_store.DB_FILE)
        print(f"\nCreated user '{username}' with access: {folders}")
    except ValueError as e:
        print(f"\nCouldn't create user: {e}")


def list_users_flow():
    users = auth_store.list_users()
    if not users:
        print("No users yet.")
        return
    print(f"\n{'Username':<20} {'Access':<40} Created")
    print("-" * 80)
    for u in users:
        access = "ALL" if u["folders"] == "ALL" else (", ".join(u["folders"]) or "(none)")
        print(f"{u['username']:<20} {access:<40} {u['created_at'][:10]}")


def delete_user_flow():
    list_users_flow()
    username = input("\nUsername to delete: ").strip()
    if not username:
        return
    confirm = input(f"Type '{username}' again to confirm deletion: ").strip()
    if confirm != username:
        print("Didn't match - nothing deleted.")
        return
    auth_store.delete_user(username)
    print(f"Deleted '{username}'.")


def main():
    auth_store.init_db()

    while True:
        print("\n1. Create user")
        print("2. List users")
        print("3. Delete user")
        print("4. Quit")
        choice = input("Choose an option: ").strip()

        if choice == "1":
            create_user_flow()
        elif choice == "2":
            list_users_flow()
        elif choice == "3":
            delete_user_flow()
        elif choice == "4":
            break
        else:
            print("Not a valid option - type 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()