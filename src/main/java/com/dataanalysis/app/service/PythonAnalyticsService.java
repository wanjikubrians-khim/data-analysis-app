package com.dataanalysis.app.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.Map;

@Service
public class PythonAnalyticsService {

    private static final Logger logger = LoggerFactory.getLogger(PythonAnalyticsService.class);

    @Value("${python.analytics.url:http://localhost:5000}")
    private String pythonApiUrl;

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;

    public PythonAnalyticsService() {
        this.restTemplate = new RestTemplate();
        this.objectMapper = new ObjectMapper();
    }

    /**
     * Check if Python analytics service is available
     */
    public boolean isPythonServiceAvailable() {
        try {
            ResponseEntity<String> response = restTemplate.getForEntity(
                pythonApiUrl + "/health", String.class);
            return response.getStatusCode() == HttpStatus.OK;
        } catch (Exception e) {
            logger.warn("Python analytics service not available: {}", e.getMessage());
            return false;
        }
    }

    /**
     * Perform comprehensive Python-based analysis
     */
    public Map<String, Object> performComprehensiveAnalysis(MultipartFile file, 
                                                          String targetColumn, 
                                                          Integer nClusters) throws IOException {
        if (!isPythonServiceAvailable()) {
            return createErrorResponse("Python analytics service is not available. Please ensure the Python server is running on " + pythonApiUrl);
        }

        try {
            // Prepare multipart request
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            
            // Add file
            body.add("file", new MultipartFileResource(file));
            
            // Add optional parameters
            if (targetColumn != null && !targetColumn.trim().isEmpty()) {
                body.add("target_column", targetColumn);
            }
            if (nClusters != null && nClusters > 0) {
                body.add("n_clusters", nClusters.toString());
            }

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            // Make request to Python API
            ResponseEntity<String> response = restTemplate.postForEntity(
                pythonApiUrl + "/api/analyze/comprehensive", requestEntity, String.class);

            if (response.getStatusCode() == HttpStatus.OK) {
                JsonNode jsonResponse = objectMapper.readTree(response.getBody());
                return objectMapper.convertValue(jsonResponse, Map.class);
            } else {
                return createErrorResponse("Python analysis failed with status: " + response.getStatusCode());
            }

        } catch (Exception e) {
            logger.error("Error performing comprehensive analysis", e);
            return createErrorResponse("Analysis failed: " + e.getMessage());
        }
    }

    /**
     * Perform machine learning analysis
     */
    public Map<String, Object> performMachineLearningAnalysis(MultipartFile file, String targetColumn) throws IOException {
        if (!isPythonServiceAvailable()) {
            return createErrorResponse("Python analytics service is not available");
        }

        if (targetColumn == null || targetColumn.trim().isEmpty()) {
            return createErrorResponse("Target column is required for machine learning analysis");
        }

        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new MultipartFileResource(file));
            body.add("target_column", targetColumn);

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            ResponseEntity<String> response = restTemplate.postForEntity(
                pythonApiUrl + "/api/analyze/machine_learning", requestEntity, String.class);

            if (response.getStatusCode() == HttpStatus.OK) {
                JsonNode jsonResponse = objectMapper.readTree(response.getBody());
                return objectMapper.convertValue(jsonResponse, Map.class);
            } else {
                return createErrorResponse("Machine learning analysis failed with status: " + response.getStatusCode());
            }

        } catch (Exception e) {
            logger.error("Error performing ML analysis", e);
            return createErrorResponse("ML analysis failed: " + e.getMessage());
        }
    }

    /**
     * Perform correlation analysis
     */
    public Map<String, Object> performCorrelationAnalysis(MultipartFile file) throws IOException {
        if (!isPythonServiceAvailable()) {
            return createErrorResponse("Python analytics service is not available");
        }

        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new MultipartFileResource(file));

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            ResponseEntity<String> response = restTemplate.postForEntity(
                pythonApiUrl + "/api/analyze/correlation", requestEntity, String.class);

            if (response.getStatusCode() == HttpStatus.OK) {
                JsonNode jsonResponse = objectMapper.readTree(response.getBody());
                return objectMapper.convertValue(jsonResponse, Map.class);
            } else {
                return createErrorResponse("Correlation analysis failed with status: " + response.getStatusCode());
            }

        } catch (Exception e) {
            logger.error("Error performing correlation analysis", e);
            return createErrorResponse("Correlation analysis failed: " + e.getMessage());
        }
    }

    /**
     * Perform clustering analysis
     */
    public Map<String, Object> performClusteringAnalysis(MultipartFile file, Integer nClusters) throws IOException {
        if (!isPythonServiceAvailable()) {
            return createErrorResponse("Python analytics service is not available");
        }

        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new MultipartFileResource(file));
            
            if (nClusters != null && nClusters > 0) {
                body.add("n_clusters", nClusters.toString());
            } else {
                body.add("n_clusters", "3"); // Default value
            }

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            ResponseEntity<String> response = restTemplate.postForEntity(
                pythonApiUrl + "/api/analyze/clustering", requestEntity, String.class);

            if (response.getStatusCode() == HttpStatus.OK) {
                JsonNode jsonResponse = objectMapper.readTree(response.getBody());
                return objectMapper.convertValue(jsonResponse, Map.class);
            } else {
                return createErrorResponse("Clustering analysis failed with status: " + response.getStatusCode());
            }

        } catch (Exception e) {
            logger.error("Error performing clustering analysis", e);
            return createErrorResponse("Clustering analysis failed: " + e.getMessage());
        }
    }

    /**
     * Get available models and analysis types
     */
    public Map<String, Object> getAvailableModels() {
        if (!isPythonServiceAvailable()) {
            return createErrorResponse("Python analytics service is not available");
        }

        try {
            ResponseEntity<String> response = restTemplate.getForEntity(
                pythonApiUrl + "/api/models/available", String.class);

            if (response.getStatusCode() == HttpStatus.OK) {
                JsonNode jsonResponse = objectMapper.readTree(response.getBody());
                return objectMapper.convertValue(jsonResponse, Map.class);
            } else {
                return createErrorResponse("Failed to get available models");
            }

        } catch (Exception e) {
            logger.error("Error getting available models", e);
            return createErrorResponse("Failed to get available models: " + e.getMessage());
        }
    }

    /**
     * Generate sample data and analysis
     */
    public Map<String, Object> generateSampleAnalysis() {
        if (!isPythonServiceAvailable()) {
            return createErrorResponse("Python analytics service is not available");
        }

        try {
            ResponseEntity<String> response = restTemplate.getForEntity(
                pythonApiUrl + "/api/sample/generate", String.class);

            if (response.getStatusCode() == HttpStatus.OK) {
                JsonNode jsonResponse = objectMapper.readTree(response.getBody());
                return objectMapper.convertValue(jsonResponse, Map.class);
            } else {
                return createErrorResponse("Failed to generate sample analysis");
            }

        } catch (Exception e) {
            logger.error("Error generating sample analysis", e);
            return createErrorResponse("Failed to generate sample analysis: " + e.getMessage());
        }
    }

    private Map<String, Object> createErrorResponse(String message) {
        Map<String, Object> errorResponse = new HashMap<>();
        errorResponse.put("error", true);
        errorResponse.put("message", message);
        errorResponse.put("python_service_available", isPythonServiceAvailable());
        return errorResponse;
    }

    /**
     * Custom MultipartFile resource for RestTemplate
     */
    private static class MultipartFileResource extends FileSystemResource {
        private final MultipartFile multipartFile;

        public MultipartFileResource(MultipartFile multipartFile) throws IOException {
            super(createTempFile(multipartFile));
            this.multipartFile = multipartFile;
        }

        private static File createTempFile(MultipartFile multipartFile) throws IOException {
            Path tempFile = Files.createTempFile("upload_", "_" + multipartFile.getOriginalFilename());
            Files.copy(multipartFile.getInputStream(), tempFile, StandardCopyOption.REPLACE_EXISTING);
            File file = tempFile.toFile();
            file.deleteOnExit(); // Clean up temp file
            return file;
        }

        @Override
        public String getFilename() {
            return multipartFile.getOriginalFilename();
        }
    }
}
