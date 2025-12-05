-- MariaDB Setup Script
-- Run this in MySQL/MariaDB as root user:
--   mysql -u root -p < setup_mariadb.sql

-- Create database
CREATE DATABASE IF NOT EXISTS makerspace_rag 
    CHARACTER SET utf8mb4 
    COLLATE utf8mb4_unicode_ci;

-- Create user
CREATE USER IF NOT EXISTS 'makerspace'@'localhost' 
    IDENTIFIED BY 'makerspace2024';

-- Create user for remote access (from any host)
CREATE USER IF NOT EXISTS 'makerspace'@'%' 
    IDENTIFIED BY 'makerspace2024';

-- Grant privileges
GRANT ALL PRIVILEGES ON makerspace_rag.* TO 'makerspace'@'localhost';
GRANT ALL PRIVILEGES ON makerspace_rag.* TO 'makerspace'@'%';

-- Apply changes
FLUSH PRIVILEGES;

-- Verify
SELECT 'Database and user created successfully!' AS Status;
SHOW DATABASES LIKE 'makerspace%';
