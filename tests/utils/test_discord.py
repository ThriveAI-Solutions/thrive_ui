#!/usr/bin/env python3
"""
Quick test script to verify Discord webhook functionality
"""
import requests

def test_discord_webhook():
    webhook_url = "https://discord.com/api/webhooks/1397887042863759360/arOUB5h6GNegNBEuXkfSvNrbGRGBBa1N9FQ_-PrfarreN5_ShqkJOciaGDg9XoUWrw-6"
    
    payload = {
        "username": "ThriveAI Test Bot",
        "embeds": [{
            "title": "TEST - Discord Webhook Test",
            "description": "This is a test message to verify Discord webhook connectivity",
            "color": 0xFF0000,  # Red
            "fields": [
                {
                    "name": "User",
                    "value": "test_user",
                    "inline": True
                },
                {
                    "name": "Module", 
                    "value": "test_discord.py",
                    "inline": True
                }
            ]
        }]
    }
    
    try:
        print(f"Testing Discord webhook: {webhook_url[:50]}...")
        response = requests.post(webhook_url, json=payload, timeout=10)
        print(f"Response status: {response.status_code}")
        if response.status_code != 204:
            print(f"Response text: {response.text}")
        response.raise_for_status()
        print("✅ Discord webhook test successful!")
        return True
    except Exception as e:
        print(f"❌ Discord webhook test failed: {e}")
        return False

if __name__ == "__main__":
    test_discord_webhook()