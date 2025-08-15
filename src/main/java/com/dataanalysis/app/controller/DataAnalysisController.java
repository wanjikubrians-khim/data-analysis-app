package com.dataanalysis.app.controller;

import com.dataanalysis.app.service.DataAnalysisService;
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
}
