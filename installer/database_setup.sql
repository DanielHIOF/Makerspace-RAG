-- Makerspace RAG - Database Setup Script
-- Run this script to set up the database and remote access user

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS makerspace_rag
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

-- Create local user (for same machine)
CREATE USER IF NOT EXISTS 'makerspace'@'localhost'
    IDENTIFIED BY 'makerspace2024';

GRANT ALL PRIVILEGES ON makerspace_rag.* TO 'makerspace'@'localhost';

-- Create remote user (for network access)
-- '%' allows connections from any IP - restrict to specific IP for security
CREATE USER IF NOT EXISTS 'makerspace'@'%'
    IDENTIFIED BY 'makerspace2024';

GRANT ALL PRIVILEGES ON makerspace_rag.* TO 'makerspace'@'%';

-- Apply privileges
FLUSH PRIVILEGES;

-- Verify setup
SELECT
    user,
    host,
    plugin
FROM mysql.user
WHERE user = 'makerspace';

-- Show databases
SHOW DATABASES LIKE 'makerspace%';
