#!/bin/bash

# OpenAI API 키 설정 스크립트
echo "OpenAI API 키를 입력해주세요:"
read -s api_key

if [ ! -z "$api_key" ]; then
    export OPENAI_API_KEY="$api_key"
    echo "✅ API 키가 설정되었습니다."
    echo "현재 세션에서만 유효합니다."
    echo ""
    echo "영구 설정을 원한다면 ~/.zshrc 또는 ~/.bashrc에 다음 줄을 추가하세요:"
    echo "export OPENAI_API_KEY=\"$api_key\""
else
    echo "❌ API 키가 입력되지 않았습니다."
fi
