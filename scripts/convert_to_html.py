#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
마크다운을 HTML로 변환하는 스크립트
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import markdown
from pathlib import Path
from src.data import PATHS

def convert_markdown_to_html():
    """마크다운 보고서를 HTML로 변환합니다."""
    
    # 마크다운 파일 경로
    md_path = PATHS.reports / "STR_Report.md"
    
    if not md_path.exists():
        print(f"❌ 마크다운 파일을 찾을 수 없습니다: {md_path}")
        return
    
    # 마크다운 내용 읽기
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # HTML 헤더와 스타일
    html_header = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>의심거래 분석(STR) 보고서</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            text-align: center;
        }
        h2 {
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }
        h3 {
            color: #2c3e50;
            margin-top: 25px;
        }
        h4 {
            color: #34495e;
            background-color: #ecf0f1;
            padding: 10px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .transaction-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .feature-list {
            background: #e8f4fd;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }
        .feature-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #d1ecf1;
        }
        .feature-item:last-child {
            border-bottom: none;
        }
        .shap-value {
            font-weight: bold;
            color: #e74c3c;
        }
        .raw-value {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        .contribution-positive {
            color: #27ae60;
        }
        .contribution-negative {
            color: #e74c3c;
        }
        .status-correct {
            color: #27ae60;
            font-weight: bold;
        }
        .status-incorrect {
            color: #e74c3c;
            font-weight: bold;
        }
        .performance-summary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .performance-item {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        .performance-item:last-child {
            border-bottom: none;
        }
        .shap-importance {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
        }
        .llm-explanation {
            background: #f1f8e9;
            border: 1px solid #c5e1a5;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin: 15px 0;
        }
        .emoji {
            font-size: 1.2em;
        }
        ul {
            padding-left: 20px;
        }
        li {
            margin: 8px 0;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background: #ecf0f1;
            border-radius: 8px;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="container">
"""
    
    html_footer = """
    </div>
    <div class="footer">
        <p>보고서 생성 완료 ✅ | XAI 시스템</p>
    </div>
</body>
</html>
"""
    
    # 마크다운을 HTML로 변환
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'codehilite'])
    html_content = md.convert(md_content)
    
    # HTML 파일 저장
    html_path = PATHS.reports / "STR_Report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_header + html_content + html_footer)
    
    print(f"✅ HTML 보고서 생성 완료: {html_path}")
    print(f"🌐 웹 브라우저에서 열어보세요: {html_path}")
    
    return html_path

if __name__ == "__main__":
    convert_markdown_to_html()
