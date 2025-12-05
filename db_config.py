# Database Configuration
# Edit these values for your environment

# Local development (Windows)
DB_HOST = 'localhost'
DB_PORT = 3306
DB_USER = 'makerspace'
DB_PASSWORD = 'makerspace2024'  # Change this!
DB_NAME = 'makerspace_rag'

# For RPI4 deployment, change DB_HOST to the Pi's IP
# DB_HOST = '192.168.1.100'

# Build connection string
DATABASE_URI = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Fallback to SQLite if needed (for testing without MariaDB)
# DATABASE_URI = 'sqlite:///makerspace.db'
