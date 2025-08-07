#!/usr/bin/env python3
"""
Debug script for Gemini configuration issues.
Run this to check your secrets configuration and clear caches.
"""

import logging
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def check_secrets():
    """Check if secrets are properly configured."""
    try:
        import streamlit as st

        print("=== SECRETS CHECK ===")

        # Check if secrets file exists
        secrets_file = Path(".streamlit/secrets.toml")
        if not secrets_file.exists():
            print("❌ secrets.toml file not found!")
            print("Please copy .streamlit/secrets_example.toml to .streamlit/secrets.toml")
            return False

        print("✅ secrets.toml file exists")

        # Try to access secrets
        try:
            ai_keys = dict(st.secrets.get("ai_keys", {}))
            rag_model = dict(st.secrets.get("rag_model", {}))

            print(f"📋 AI Keys available: {list(ai_keys.keys())}")
            print(f"📋 RAG Model config: {rag_model}")

            # Check Gemini specific config
            if "gemini_model" in ai_keys:
                print(f"✅ gemini_model: {ai_keys['gemini_model']}")
            else:
                print("❌ gemini_model not found in secrets")

            if "gemini_api" in ai_keys:
                print("✅ gemini_api: [CONFIGURED]")
            else:
                print("❌ gemini_api not found in secrets")

            if "chroma_path" in rag_model:
                print(f"✅ chroma_path: {rag_model['chroma_path']}")
            else:
                print("❌ chroma_path not found in rag_model")
                print("Add this to your secrets.toml:")
                print("[rag_model]")
                print('chroma_path = "./chromadb"')

            return True

        except Exception as e:
            print(f"❌ Error accessing secrets: {e}")
            return False

    except ImportError:
        print("❌ Streamlit not available. Run this from your project environment.")
        return False


def clear_streamlit_cache():
    """Clear Streamlit cache directory."""
    import shutil

    print("\n=== CACHE CLEARING ===")

    # Common Streamlit cache locations
    cache_dirs = [
        Path.home() / ".streamlit",
        Path(".streamlit") / "cache",
        Path("cache"),
    ]

    for cache_dir in cache_dirs:
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                print(f"✅ Cleared cache: {cache_dir}")
            except Exception as e:
                print(f"❌ Failed to clear {cache_dir}: {e}")
        else:
            print(f"ℹ️  Cache dir not found: {cache_dir}")


def test_vanna_service():
    """Test VannaService initialization."""
    try:
        print("\n=== VANNA SERVICE TEST ===")

        # Use the new force cache clearing method
        from utils.vanna_calls import VannaService

        VannaService.force_cache_clear()

        # Try to create a VannaService instance
        from utils.vanna_calls import UserContext, extract_vanna_config_from_secrets

        user_context = UserContext(user_id="debug_user", user_role=1)
        config = extract_vanna_config_from_secrets()

        print("📋 Creating VannaService instance...")
        service = VannaService.get_instance(user_context, config)

        if service:
            print("✅ VannaService created successfully")
            print(f"📋 Service type: {type(service.vn).__name__}")
            return True
        else:
            print("❌ Failed to create VannaService")
            return False

    except Exception as e:
        print(f"❌ Error testing VannaService: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main debug function."""
    setup_logging()

    print("🔍 Debugging Gemini Configuration Issues")
    print("=" * 50)

    # Step 1: Check secrets
    if not check_secrets():
        print("\n❌ Configuration issues found. Please fix your secrets.toml file.")
        return

    # Step 2: Clear caches
    clear_streamlit_cache()

    # Step 3: Test VannaService
    if test_vanna_service():
        print("\n✅ Configuration appears to be working!")
        print("\n📋 Next steps:")
        print("1. Run your Streamlit app")
        print("2. Check the terminal logs for the debug messages")
        print("3. Look for '=== VANNA SETUP DEBUG ===' in the logs")
    else:
        print("\n❌ VannaService test failed. Check the error messages above.")

    print("\n🔧 Troubleshooting tips:")
    print("1. Make sure your .streamlit/secrets.toml has:")
    print("   [ai_keys]")
    print('   gemini_model = "gemini-2.0-flash-exp"')
    print('   gemini_api = "your_api_key"')
    print("   [rag_model]")
    print('   chroma_path = "./chromadb"')
    print("\n2. Restart your Streamlit app completely")
    print("3. Check the logs for the debug messages we added")


if __name__ == "__main__":
    main()
