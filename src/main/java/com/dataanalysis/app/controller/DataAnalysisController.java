package com.dataanalysis.app.controller;

import com.dataanalysis.app.service.DataAnalysisService;
import com.dataanalysis.app.service.PythonAnalyticsService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.util.Map;

@Controller
public class DataAnalysisController {

    @Autowired
    private DataAnalysisService dataAnalysisService;
    
    @Autowired
    private PythonAnalyticsService pythonAnalyticsService;

    @GetMapping("/")
    public String home() {
        return "index";
    }

    @GetMapping("/upload")
    public String uploadPage() {
        return "upload";
    }

    @PostMapping("/upload")
    public String uploadFile(@RequestParam("file") MultipartFile file, 
                           RedirectAttributes redirectAttributes, 
                           Model model) {
        if (file.isEmpty()) {
            redirectAttributes.addFlashAttribute("error", "Please select a file to upload");
            return "redirect:/upload";
        }

        try {
            Map<String, Object> analysisResult = dataAnalysisService.analyzeFile(file);
            model.addAttribute("analysisResult", analysisResult);
            model.addAttribute("filename", file.getOriginalFilename());
            return "results";
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Error analyzing file: " + e.getMessage());
            return "redirect:/upload";
        }
    }

    @GetMapping("/sample-data")
    public String sampleData(Model model) {
        try {
            Map<String, Object> sampleAnalysis = dataAnalysisService.generateSampleAnalysis();
            model.addAttribute("analysisResult", sampleAnalysis);
            model.addAttribute("filename", "Sample Dataset");
            return "results";
        } catch (Exception e) {
            model.addAttribute("error", "Error generating sample data: " + e.getMessage());
            return "index";
        }
    }

    @GetMapping("/api/analysis/{type}")
    @ResponseBody
    public Map<String, Object> getAnalysisData(@PathVariable String type) {
        return dataAnalysisService.getAnalysisByType(type);
    }
    
    // Python Analytics Endpoints
    
    @GetMapping("/python-analytics")
    public String pythonAnalyticsPage(Model model) {
        model.addAttribute("pythonServiceAvailable", pythonAnalyticsService.isPythonServiceAvailable());
        return "python-analytics";
    }
    
    @PostMapping("/api/python/comprehensive")
    @ResponseBody
    public Map<String, Object> pythonComprehensiveAnalysis(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "target_column", required = false) String targetColumn,
            @RequestParam(value = "n_clusters", required = false) Integer nClusters) {
        try {
            return pythonAnalyticsService.performComprehensiveAnalysis(file, targetColumn, nClusters);
        } catch (Exception e) {
            return Map.of("error", true, "message", "Analysis failed: " + e.getMessage());
        }
    }
    
    @PostMapping("/api/python/ml")
    @ResponseBody
    public Map<String, Object> pythonMachineLearning(
            @RequestParam("file") MultipartFile file,
            @RequestParam("target_column") String targetColumn) {
        try {
            return pythonAnalyticsService.performMachineLearningAnalysis(file, targetColumn);
        } catch (Exception e) {
            return Map.of("error", true, "message", "ML analysis failed: " + e.getMessage());
        }
    }
    
    @PostMapping("/api/python/correlation")
    @ResponseBody
    public Map<String, Object> pythonCorrelationAnalysis(@RequestParam("file") MultipartFile file) {
        try {
            return pythonAnalyticsService.performCorrelationAnalysis(file);
        } catch (Exception e) {
            return Map.of("error", true, "message", "Correlation analysis failed: " + e.getMessage());
        }
    }
    
    @PostMapping("/api/python/clustering")
    @ResponseBody
    public Map<String, Object> pythonClusteringAnalysis(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "n_clusters", required = false) Integer nClusters) {
        try {
            return pythonAnalyticsService.performClusteringAnalysis(file, nClusters);
        } catch (Exception e) {
            return Map.of("error", true, "message", "Clustering analysis failed: " + e.getMessage());
        }
    }
    
    @GetMapping("/api/python/models")
    @ResponseBody
    public Map<String, Object> getPythonModels() {
        return pythonAnalyticsService.getAvailableModels();
    }
    
    @GetMapping("/api/python/sample")
    @ResponseBody
    public Map<String, Object> getPythonSampleAnalysis() {
        return pythonAnalyticsService.generateSampleAnalysis();
    }
    
    @GetMapping("/api/python/health")
    @ResponseBody
    public Map<String, Object> checkPythonHealth() {
        boolean available = pythonAnalyticsService.isPythonServiceAvailable();
        return Map.of(
            "python_service_available", available,
            "status", available ? "healthy" : "unavailable",
            "message", available ? "Python analytics service is running" : "Python analytics service is not available"
        );
    }
}
