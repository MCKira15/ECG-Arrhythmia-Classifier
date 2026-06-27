CREATE DATABASE IF NOT EXISTS ecg_arrhythmia
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'ecg_user'@'%' IDENTIFIED BY 'ecg_password';

GRANT ALL PRIVILEGES ON ecg_arrhythmia.* TO 'ecg_user'@'%';

FLUSH PRIVILEGES;
