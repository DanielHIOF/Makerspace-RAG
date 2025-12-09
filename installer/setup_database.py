"""
Makerspace RAG - Database Setup Script
Configures MariaDB for remote access
"""

import os
import sys
import subprocess
import socket
from pathlib import Path

# MariaDB paths (adjust if needed)
MARIADB_PATHS = [
    r"C:\Program Files\MariaDB 12.1",
    r"C:\Program Files\MariaDB 11.0",
    r"C:\Program Files\MariaDB 10.11",
    r"C:\Program Files (x86)\MariaDB",
]

SCRIPT_DIR = Path(__file__).parent


def print_status(message, status="INFO"):
    """Print formatted status message."""
    icons = {"INFO": "i", "OK": "+", "WARN": "!", "ERROR": "x", "WAIT": "~"}
    icon = icons.get(status, "*")
    print(f"  [{icon}] {message}")


def find_mariadb():
    """Find MariaDB installation directory."""
    for path in MARIADB_PATHS:
        if os.path.exists(path):
            return Path(path)

    # Try to find via where command
    try:
        result = subprocess.run(['where', 'mysql'], capture_output=True, text=True)
        if result.returncode == 0:
            mysql_path = Path(result.stdout.strip().split('\n')[0])
            return mysql_path.parent.parent
    except:
        pass

    return None


def get_mariadb_config_path(mariadb_dir):
    """Get path to MariaDB config file."""
    possible_paths = [
        mariadb_dir / 'data' / 'my.ini',
        mariadb_dir / 'my.ini',
        Path(r'C:\ProgramData\MySQL\MySQL Server 8.0\my.ini'),
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return mariadb_dir / 'data' / 'my.ini'


def get_local_ip():
    """Get local IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "unknown"


def check_mariadb_running():
    """Check if MariaDB service is running."""
    try:
        result = subprocess.run(
            ['sc', 'query', 'MariaDB'],
            capture_output=True,
            text=True
        )
        return 'RUNNING' in result.stdout
    except:
        return False


def configure_remote_access(mariadb_dir):
    """Configure MariaDB for remote access."""
    config_path = get_mariadb_config_path(mariadb_dir)

    print_status(f"MariaDB config: {config_path}", "INFO")

    if not config_path.exists():
        print_status("Config file not found - creating new one", "WARN")
        config_content = """[mysqld]
# Allow remote connections
bind-address = 0.0.0.0
port = 3306

# Performance settings
innodb_buffer_pool_size = 256M
max_connections = 100

# Character set
character-set-server = utf8mb4
collation-server = utf8mb4_unicode_ci
"""
    else:
        # Read existing config
        with open(config_path, 'r', encoding='utf-8', errors='ignore') as f:
            config_content = f.read()

        # Check if bind-address is already set
        if 'bind-address' in config_content:
            if '0.0.0.0' in config_content or '::' in config_content:
                print_status("Remote access already configured", "OK")
                return True
            else:
                # Update bind-address
                lines = config_content.split('\n')
                new_lines = []
                for line in lines:
                    if line.strip().startswith('bind-address'):
                        new_lines.append('bind-address = 0.0.0.0')
                        print_status("Updated bind-address to 0.0.0.0", "OK")
                    else:
                        new_lines.append(line)
                config_content = '\n'.join(new_lines)
        else:
            # Add bind-address after [mysqld]
            if '[mysqld]' in config_content:
                config_content = config_content.replace(
                    '[mysqld]',
                    '[mysqld]\nbind-address = 0.0.0.0'
                )
                print_status("Added bind-address = 0.0.0.0", "OK")
            else:
                config_content = '[mysqld]\nbind-address = 0.0.0.0\n\n' + config_content
                print_status("Added [mysqld] section with bind-address", "OK")

    # Write config
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print_status(f"Config saved to {config_path}", "OK")
        return True
    except PermissionError:
        print_status("Permission denied - run as Administrator", "ERROR")
        return False
    except Exception as e:
        print_status(f"Failed to write config: {e}", "ERROR")
        return False


def run_sql_script(mariadb_dir, sql_file, root_password=None):
    """Run SQL script against MariaDB."""
    mysql_path = mariadb_dir / 'bin' / 'mysql.exe'

    if not mysql_path.exists():
        print_status(f"mysql.exe not found at {mysql_path}", "ERROR")
        return False

    cmd = [str(mysql_path), '-u', 'root']
    if root_password:
        cmd.extend(['-p' + root_password])

    try:
        with open(sql_file, 'r') as f:
            sql_content = f.read()

        result = subprocess.run(
            cmd,
            input=sql_content,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print_status("SQL script executed successfully", "OK")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print_status(f"SQL error: {result.stderr}", "ERROR")
            return False

    except Exception as e:
        print_status(f"Failed to run SQL: {e}", "ERROR")
        return False


def configure_firewall():
    """Add firewall rule for MariaDB."""
    print_status("Configuring Windows Firewall...", "WAIT")

    try:
        # Check if rule already exists
        result = subprocess.run(
            ['netsh', 'advfirewall', 'firewall', 'show', 'rule', 'name=MariaDB'],
            capture_output=True,
            text=True
        )

        if 'MariaDB' in result.stdout:
            print_status("Firewall rule already exists", "OK")
            return True

        # Add firewall rule
        result = subprocess.run([
            'netsh', 'advfirewall', 'firewall', 'add', 'rule',
            'name=MariaDB',
            'dir=in',
            'action=allow',
            'protocol=tcp',
            'localport=3306'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print_status("Firewall rule added for port 3306", "OK")
            return True
        else:
            print_status(f"Failed to add firewall rule: {result.stderr}", "ERROR")
            return False

    except Exception as e:
        print_status(f"Firewall configuration failed: {e}", "ERROR")
        return False


def restart_mariadb():
    """Restart MariaDB service."""
    print_status("Restarting MariaDB service...", "WAIT")

    try:
        subprocess.run(['net', 'stop', 'MariaDB'], capture_output=True)
        subprocess.run(['net', 'start', 'MariaDB'], capture_output=True)
        print_status("MariaDB restarted", "OK")
        return True
    except Exception as e:
        print_status(f"Failed to restart MariaDB: {e}", "ERROR")
        return False


def main():
    print("\n" + "="*60)
    print("  Makerspace RAG - Database Remote Access Setup")
    print("="*60 + "\n")

    # Check if running as admin
    try:
        is_admin = os.getuid() == 0
    except AttributeError:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0

    if not is_admin:
        print_status("WARNING: Run as Administrator for full setup", "WARN")

    # Find MariaDB
    mariadb_dir = find_mariadb()
    if not mariadb_dir:
        print_status("MariaDB installation not found", "ERROR")
        print_status("Please install MariaDB from https://mariadb.org/download/", "INFO")
        return 1

    print_status(f"Found MariaDB at: {mariadb_dir}", "OK")

    # Check if running
    if check_mariadb_running():
        print_status("MariaDB service is running", "OK")
    else:
        print_status("MariaDB service not running - starting...", "WARN")
        subprocess.run(['net', 'start', 'MariaDB'], capture_output=True)

    # Configure remote access
    print("\n[Step 1] Configuring remote access...")
    if not configure_remote_access(mariadb_dir):
        print_status("Failed to configure remote access", "ERROR")

    # Run SQL setup script
    print("\n[Step 2] Setting up database and users...")
    sql_file = SCRIPT_DIR / 'database_setup.sql'
    if sql_file.exists():
        print_status("Enter MariaDB root password (or press Enter if none):")
        root_password = input("  Password: ").strip()
        run_sql_script(mariadb_dir, sql_file, root_password if root_password else None)
    else:
        print_status(f"SQL file not found: {sql_file}", "WARN")

    # Configure firewall
    print("\n[Step 3] Configuring firewall...")
    configure_firewall()

    # Restart MariaDB
    print("\n[Step 4] Restarting MariaDB...")
    restart_mariadb()

    # Summary
    local_ip = get_local_ip()
    print("\n" + "="*60)
    print("  Setup Complete!")
    print("="*60)
    print(f"""
  Database: makerspace_rag
  User: makerspace
  Password: makerspace2024
  Port: 3306

  Local connection:
    DB_HOST=localhost

  Remote connection (from other machines):
    DB_HOST={local_ip}

  To connect from another PC, use:
    mysql -h {local_ip} -u makerspace -p makerspace_rag

  Don't forget to:
  1. Set DB_HOST in .env file on remote machines
  2. Ensure firewall allows port 3306
  3. Check network connectivity
""")

    return 0


if __name__ == '__main__':
    sys.exit(main())
