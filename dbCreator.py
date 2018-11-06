# -*- encoding: utf-8 -*-

import struct
import binascii

import mysql.connector

create_db_statement = "CREATE DATABASE IF NOT EXISTS `test` /*!40100 COLLATE 'utf8_general_ci' */"

createStatements = list()


createStatements.append(
    "CREATE TABLE IF NOT EXISTS `tb_sensor` ("
        "`id` INT(11) NOT NULL AUTO_INCREMENT, "
        "`type` INT(11) NOT NULL DEFAULT '0', "
        "`transformation_matrix` LONGTEXT NOT NULL DEFAULT '0' COLLATE 'utf8_general_ci', "
        "PRIMARY KEY (`id`)"
    ") "
    "COLLATE='utf8_general_ci' "
    "ENGINE=InnoDB"
    ";")



createStatements.append(
    "CREATE TABLE IF NOT EXISTS `tb_sample_flag` ("
        "`id` INT(11) NOT NULL AUTO_INCREMENT, "
        "`name` VARCHAR(1000) NOT NULL DEFAULT '', "
        "PRIMARY KEY (`id`) "
    ") "
    "ENGINE=InnoDB "
    ";")




tb_rsds_sample_CREATE_STATEMENT_TEMPLATE = (
    "CREATE TABLE IF NOT EXISTS `tb_rsds_sample` ("
        "`id` INT UNSIGNED NOT NULL AUTO_INCREMENT, "
        "`sensor_id` INT UNSIGNED NOT NULL, "
        "`sequence_id` INT UNSIGNED NOT NULL, "
        "`timestamp` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, "
        "`sequence_time_ms` BIGINT UNSIGNED NOT NULL, "
        "`byte_type` BLOB NOT NULL DEFAULT '0', "
        # "{}, "
    "PRIMARY KEY (`id`), "
    "UNIQUE INDEX `unique_index` (`id`, `sensor_id`, `sequence_id`, `timestamp`, `sequence_time_ms`) "
    ") "
    "COLLATE='utf8_general_ci' "
    "ENGINE=InnoDB"
    ";")

# rsdsPointCount = 3
# dataColumns = list()
# for i in range(rsdsPointCount):
#     dataColumns.append("`angle_{}` FLOAT NOT NULL".format(i))
#     dataColumns.append("`range_{}` FLOAT NOT NULL".format(i))
#     dataColumns.append("`amplitude_{}` FLOAT NOT NULL".format(i))
#     dataColumns.append("`doppler_{}` FLOAT NOT NULL".format(i))
# dataColumns = ", ".join(dataColumns)
# createStatements.append(tb_rsds_sample_CREATE_STATEMENT_TEMPLATE.format(dataColumns))

createStatements.append(tb_rsds_sample_CREATE_STATEMENT_TEMPLATE)





createStatements.append(
    "CREATE TABLE IF NOT EXISTS `tb_valeo_scala_sample` ("
    "    `id` INT UNSIGNED NOT NULL AUTO_INCREMENT, "
    "    `sensor_id` INT UNSIGNED NOT NULL, "
    "    `sequence_id` INT UNSIGNED NOT NULL, "
    "    `timestamp` TIME NOT NULL, "
    "    `sequence_time_ms` BIGINT UNSIGNED NOT NULL, "
    "    `data` LONGTEXT NOT NULL DEFAULT '0' COLLATE 'utf8mb4_bin', "
    "PRIMARY KEY (`id`), "
    "UNIQUE INDEX `unique_index` (`id`, `sensor_id`, `sequence_id`, `timestamp`, `sequence_time_ms`) "
    ") "
    "COLLATE='utf8_general_ci' "
    "ENGINE=InnoDB"
    ";")


createStatements.append(
    "CREATE TABLE IF NOT EXISTS `tb_ibeo_fusion_objects_sample` ("
    "    `id` INT UNSIGNED NOT NULL AUTO_INCREMENT, "
    "    `sensor_id` INT UNSIGNED NOT NULL, "
    "    `sequence_id` INT UNSIGNED NOT NULL, "
    "    `timestamp` TIME NOT NULL, "
    "    `sequence_time_ms` BIGINT UNSIGNED NOT NULL, "
    "    `data` LONGTEXT NOT NULL DEFAULT '0' COLLATE 'utf8mb4_bin', "
    "PRIMARY KEY (`id`), "
    "UNIQUE INDEX `unique_index` (`id`, `sensor_id`, `sequence_id`, `timestamp`, `sequence_time_ms`) "
    ") "
    "COLLATE='utf8_general_ci' "
    "ENGINE=InnoDB"
    ";")



createStatements.append(
    "CREATE TABLE IF NOT EXISTS `tb_video_sample` ("
    "    `id` INT UNSIGNED NOT NULL AUTO_INCREMENT, "
    "    `sensor_id` INT UNSIGNED NOT NULL, "
    "    `sequence_id` INT UNSIGNED NOT NULL, "
    "    `timestamp` TIME NOT NULL, "
    "    `sequence_time_ms` BIGINT UNSIGNED NOT NULL, "
    "	 `relative_path_to_frame` VARCHAR(1000) NOT NULL DEFAULT '', "
    "PRIMARY KEY (`id`), "
    "UNIQUE INDEX `unique_index` (`id`, `sensor_id`, `sequence_id`, `timestamp`, `sequence_time_ms`) "
    ") "
    "COLLATE='utf8_general_ci' "
    "ENGINE=InnoDB"
    ";")


createStatements.append(
    "CREATE TABLE IF NOT EXISTS `tb_test_byte` ("
        "`id` INT UNSIGNED NOT NULL AUTO_INCREMENT, "
        "`byte_object` BLOB NOT NULL DEFAULT '0', "
    "PRIMARY KEY (`id`) "
    ") "
    "COLLATE='utf8_general_ci' "
    "ENGINE=InnoDB"
    ";")

def createDb():
    mydb = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        passwd="nntrainer",
        charset='utf8',
        use_unicode=True
    )

    mycursor = mydb.cursor()

    mycursor.execute(create_db_statement)
    mydb.commit()

    mycursor.execute("USE test")

    for createStatement in createStatements:
        mycursor.execute(createStatement)


    bytesObject = struct.pack("fff", 0.1, 0.2, 0.3)
    print(bytesObject)


    insertBytes = "INSERT INTO tb_test_byte (byte_object) VALUES (_utf8mb4 %s);"
    mycursor.execute(insertBytes, (bytesObject,))
    mydb.commit()

    getByte = "SELECT id, HEX(byte_object) FROM tb_test_byte"
    mycursor.execute(getByte)

    for row in mycursor:
        packedData = row[1]
        binaryString = binascii.unhexlify(packedData)
        unpacked = struct.unpack("fff", binaryString)
        print(unpacked)

    mydb.close()


if __name__ == "__main__":
    createDb()