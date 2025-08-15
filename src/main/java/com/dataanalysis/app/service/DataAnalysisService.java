package com.dataanalysis.app.service;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class DataAnalysisService {

    public Map<String, Object> analyzeFile(MultipartFile file) throws IOException {
        String filename = file.getOriginalFilename();
        if (filename != null && filename.toLowerCase().endsWith(".csv")) {
            return analyzeCsvFile(file);
        } else {
            throw new IllegalArgumentException("Unsupported file format. Please upload a CSV file.");
        }
    }

    private Map<String, Object> analyzeCsvFile(MultipartFile file) throws IOException {
        Map<String, Object> result = new HashMap<>();
        List<Map<String, String>> data = new ArrayList<>();
        List<String> headers = new ArrayList<>();
        Map<String, DescriptiveStatistics> numericStats = new HashMap<>();

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(file.getInputStream()));
             CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader())) {

            headers.addAll(csvParser.getHeaderNames());
            
            // Initialize statistics for numeric columns
            for (String header : headers) {
                numericStats.put(header, new DescriptiveStatistics());
            }

            for (CSVRecord record : csvParser) {
                Map<String, String> row = new HashMap<>();
                for (String header : headers) {
                    String value = record.get(header);
                    row.put(header, value);
                    
                    // Try to add to numeric statistics
                    try {
                        double numValue = Double.parseDouble(value);
                        numericStats.get(header).addValue(numValue);
                    } catch (NumberFormatException e) {
                        // Not a numeric value, ignore for statistics
                    }
                }
                data.add(row);
            }
        }

        // Compile results
        result.put("filename", file.getOriginalFilename());
        result.put("totalRows", data.size());
        result.put("totalColumns", headers.size());
        result.put("headers", headers);
        result.put("data", data.stream().limit(100).collect(Collectors.toList())); // Limit preview data
        
        // Statistical analysis
        Map<String, Map<String, Double>> statistics = new HashMap<>();
        for (String header : headers) {
            DescriptiveStatistics stats = numericStats.get(header);
            if (stats.getN() > 0) {
                Map<String, Double> columnStats = new HashMap<>();
                columnStats.put("count", (double) stats.getN());
                columnStats.put("mean", stats.getMean());
                columnStats.put("median", stats.getPercentile(50));
                columnStats.put("min", stats.getMin());
                columnStats.put("max", stats.getMax());
                columnStats.put("stddev", stats.getStandardDeviation());
                columnStats.put("variance", stats.getVariance());
                statistics.put(header, columnStats);
            }
        }
        result.put("statistics", statistics);
        
        // Generate charts data
        result.put("chartData", generateChartData(data, headers, statistics));

        return result;
    }

    public Map<String, Object> generateSampleAnalysis() {
        Map<String, Object> result = new HashMap<>();
        
        // Generate sample data
        List<Map<String, String>> sampleData = generateSampleData();
        List<String> headers = Arrays.asList("ID", "Name", "Age", "Sales", "Department");
        
        result.put("filename", "Sample Dataset");
        result.put("totalRows", sampleData.size());
        result.put("totalColumns", headers.size());
        result.put("headers", headers);
        result.put("data", sampleData);
        
        // Calculate statistics for sample data
        DescriptiveStatistics ageStats = new DescriptiveStatistics();
        DescriptiveStatistics salesStats = new DescriptiveStatistics();
        
        for (Map<String, String> row : sampleData) {
            ageStats.addValue(Double.parseDouble(row.get("Age")));
            salesStats.addValue(Double.parseDouble(row.get("Sales")));
        }
        
        Map<String, Map<String, Double>> statistics = new HashMap<>();
        
        Map<String, Double> ageStatsMap = new HashMap<>();
        ageStatsMap.put("count", (double) ageStats.getN());
        ageStatsMap.put("mean", ageStats.getMean());
        ageStatsMap.put("median", ageStats.getPercentile(50));
        ageStatsMap.put("min", ageStats.getMin());
        ageStatsMap.put("max", ageStats.getMax());
        ageStatsMap.put("stddev", ageStats.getStandardDeviation());
        statistics.put("Age", ageStatsMap);
        
        Map<String, Double> salesStatsMap = new HashMap<>();
        salesStatsMap.put("count", (double) salesStats.getN());
        salesStatsMap.put("mean", salesStats.getMean());
        salesStatsMap.put("median", salesStats.getPercentile(50));
        salesStatsMap.put("min", salesStats.getMin());
        salesStatsMap.put("max", salesStats.getMax());
        salesStatsMap.put("stddev", salesStats.getStandardDeviation());
        statistics.put("Sales", salesStatsMap);
        
        result.put("statistics", statistics);
        result.put("chartData", generateChartData(sampleData, headers, statistics));
        
        return result;
    }

    private List<Map<String, String>> generateSampleData() {
        List<Map<String, String>> data = new ArrayList<>();
        String[] names = {"John Smith", "Jane Doe", "Bob Johnson", "Alice Wilson", "Charlie Brown", 
                         "Diana Prince", "Eve Adams", "Frank Miller", "Grace Kelly", "Henry Ford"};
        String[] departments = {"Sales", "Marketing", "IT", "HR", "Finance"};
        Random random = new Random();
        
        for (int i = 1; i <= 50; i++) {
            Map<String, String> row = new HashMap<>();
            row.put("ID", String.valueOf(i));
            row.put("Name", names[random.nextInt(names.length)] + " " + i);
            row.put("Age", String.valueOf(25 + random.nextInt(40)));
            row.put("Sales", String.valueOf(1000 + random.nextInt(9000)));
            row.put("Department", departments[random.nextInt(departments.length)]);
            data.add(row);
        }
        
        return data;
    }

    private Map<String, Object> generateChartData(List<Map<String, String>> data, List<String> headers, Map<String, Map<String, Double>> statistics) {
        Map<String, Object> chartData = new HashMap<>();
        
        // Generate histogram data for numeric columns
        for (String header : headers) {
            if (statistics.containsKey(header)) {
                List<Double> values = new ArrayList<>();
                for (Map<String, String> row : data) {
                    try {
                        values.add(Double.parseDouble(row.get(header)));
                    } catch (NumberFormatException e) {
                        // Skip non-numeric values
                    }
                }
                chartData.put(header + "_histogram", values);
            }
        }
        
        return chartData;
    }

    public Map<String, Object> getAnalysisByType(String type) {
        Map<String, Object> result = new HashMap<>();
        
        switch (type.toLowerCase()) {
            case "summary":
                result.put("type", "summary");
                result.put("description", "Summary statistics and overview");
                break;
            case "correlation":
                result.put("type", "correlation");
                result.put("description", "Correlation analysis between variables");
                break;
            case "distribution":
                result.put("type", "distribution");
                result.put("description", "Distribution analysis and histograms");
                break;
            default:
                result.put("error", "Unknown analysis type: " + type);
        }
        
        return result;
    }
}
