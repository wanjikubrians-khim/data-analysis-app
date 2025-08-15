package com.dataanalysis.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.*;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.HashMap;

@Controller
public class DataAnalysisController {

    @Autowired
    private RestTemplate restTemplate;

    @Value("${python.api.url:http://localhost:5000}")
    private String pythonApiUrl;

    private final String UPLOAD_DIR = "temp_uploads";

    @GetMapping("/")
    public String index(Model model) {
        return "index";
    }

    @PostMapping("/api/upload")
    @ResponseBody
    public ResponseEntity<?> uploadFile(@RequestParam("file") MultipartFile file,
                                      @RequestParam(value = "analysisType", defaultValue = "basic") String analysisType) {
        try {
            // Create upload directory if it doesn't exist
            Path uploadPath = Paths.get(UPLOAD_DIR);
            if (!Files.exists(uploadPath)) {
                Files.createDirectories(uploadPath);
            }

            // Save the uploaded file
            String filename = System.currentTimeMillis() + "_" + file.getOriginalFilename();
            Path filePath = uploadPath.resolve(filename);
            Files.copy(file.getInputStream(), filePath);

            // Forward to Python API
            String endpoint = getPythonEndpoint(analysisType);
            ResponseEntity<Map> response = forwardToPythonAPI(filePath.toFile(), endpoint);

            // Clean up the temporary file
            Files.deleteIfExists(filePath);

            return response;

        } catch (IOException e) {
            Map<String, String> errorMap = new HashMap<>();
            errorMap.put("error", "File upload failed: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(errorMap);
        }
    }

    @PostMapping("/api/analyze")
    @ResponseBody
    public ResponseEntity<?> analyzeData(@RequestParam("file") MultipartFile file,
                                       @RequestParam(value = "analysisType", defaultValue = "comprehensive") String analysisType,
                                       @RequestParam(value = "targetColumn", required = false) String targetColumn) {
        try {
            // Create upload directory if it doesn't exist
            Path uploadPath = Paths.get(UPLOAD_DIR);
            if (!Files.exists(uploadPath)) {
                Files.createDirectories(uploadPath);
            }

            // Save the uploaded file
            String filename = System.currentTimeMillis() + "_" + file.getOriginalFilename();
            Path filePath = uploadPath.resolve(filename);
            Files.copy(file.getInputStream(), filePath);

            // Forward to Python API with analysis parameters
            String endpoint = getPythonEndpoint(analysisType);
            ResponseEntity<Map> response = forwardToPythonAPIWithParams(filePath.toFile(), endpoint, targetColumn);

            // Clean up the temporary file
            Files.deleteIfExists(filePath);

            return response;

        } catch (IOException e) {
            Map<String, String> errorMap = new HashMap<>();
            errorMap.put("error", "Analysis failed: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(errorMap);
        }
    }

    @GetMapping("/api/health")
    @ResponseBody
    public ResponseEntity<?> checkHealth() {
        try {
            // Check Python API health
            ResponseEntity<Map> response = restTemplate.getForEntity(pythonApiUrl + "/health", Map.class);
            Map<String, Object> healthMap = new HashMap<>();
            healthMap.put("java_service", "healthy");
            healthMap.put("python_service", response.getBody());
            healthMap.put("status", "all systems operational");
            return ResponseEntity.ok(healthMap);
        } catch (Exception e) {
            Map<String, Object> healthMap = new HashMap<>();
            healthMap.put("java_service", "healthy");
            healthMap.put("python_service", "unavailable");
            healthMap.put("status", "python service not available");
            healthMap.put("message", "Please ensure the Python analytics server is running on " + pythonApiUrl);
            return ResponseEntity.ok(healthMap);
        }
    }

    private String getPythonEndpoint(String analysisType) {
        switch (analysisType.toLowerCase()) {
            case "basic":
                return "/api/analyze/upload";
            case "correlation":
                return "/api/analyze/correlation";
            case "machine_learning":
            case "ml":
                return "/api/analyze/machine_learning";
            case "clustering":
                return "/api/analyze/clustering";
            case "outliers":
                return "/api/analyze/outliers";
            case "quality":
                return "/api/analyze/quality";
            case "comprehensive":
            default:
                return "/api/analyze/comprehensive";
        }
    }

    private ResponseEntity<Map> forwardToPythonAPI(File file, String endpoint) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new FileSystemResource(file));

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            return restTemplate.exchange(
                pythonApiUrl + endpoint,
                HttpMethod.POST,
                requestEntity,
                Map.class
            );
        } catch (Exception e) {
            Map<String, String> errorMap = new HashMap<>();
            errorMap.put("error", "Python API unavailable: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(errorMap);
        }
    }

    private ResponseEntity<Map> forwardToPythonAPIWithParams(File file, String endpoint, String targetColumn) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new FileSystemResource(file));
            if (targetColumn != null && !targetColumn.isEmpty()) {
                body.add("target_column", targetColumn);
            }

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            return restTemplate.exchange(
                pythonApiUrl + endpoint,
                HttpMethod.POST,
                requestEntity,
                Map.class
            );
        } catch (Exception e) {
            Map<String, String> errorMap = new HashMap<>();
            errorMap.put("error", "Python API unavailable: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(errorMap);
        }
    }
}
