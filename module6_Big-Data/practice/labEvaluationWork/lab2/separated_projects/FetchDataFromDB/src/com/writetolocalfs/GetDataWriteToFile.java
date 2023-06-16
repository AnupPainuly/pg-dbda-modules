package com.writetolocalfs;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

public class GetDataWriteToFile {

	public static void main(String[] args) {
		System.out.println("Started ..");
		ResultSet resultset = getResultSet();
		try {
			System.out.println("Writing lines into local file");
			boolean sts = writeLineIntoFile(resultset);
			if (sts) {
				System.out.println("Writing data successfully");
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("main over ..");
	}

	// get data to mysql. user table salaries
	private static ResultSet getResultSet() {
		String jdbcUrl = "jdbc:mysql://localhost:3306/test";
		String username = "root";
		String password = "cloudera";
		String query = "SELECT * FROM salaries";
		try {

			System.out.println("Loading my sql driver...");
			Class.forName("com.mysql.jdbc.Driver");
			Connection conn = DriverManager.getConnection(jdbcUrl, username,
					password);
			Statement stmt = conn.createStatement();
			ResultSet rs = stmt.executeQuery(query);
			return rs;
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

	// method write data to call getResultSet
	private static boolean writeLineIntoFile(ResultSet rs) throws SQLException {
		System.out.println("inside writeLineIntoFile");
		String filename = "/home/cloudera/shared/LoadData/LoadData/table/example.txt";
		File file = new File(filename);
		String data = "";
		try {
			// Create the file if it doesn't exist
			if (!file.exists()) {
				file.createNewFile();
				System.out.println("File created: " + file.getName());
			}
			//write to the file
			FileWriter fileWriter = new FileWriter(file.getAbsoluteFile());
			System.out.println("fileW : " + fileWriter);
			BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
			while (rs.next()) {
				data = rs.getString(1) + "," + rs.getInt(2) + ","
						+ rs.getDouble(3) + "," + rs.getInt(4) + ","
						+ rs.getInt(5);

				bufferedWriter.write(data);
				bufferedWriter.newLine();
			}
			bufferedWriter.close();
			System.out.println("Data written to the file successfully.");
			return true;
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		}
	}
}
