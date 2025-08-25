#!/usr/bin/env python3
"""
Startup script for the ink detection inference server
"""
import os
import sys
import argparse
import logging
from pathlib import Path

def setup_logging(log_level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def check_environment():
    """Check required environment variables"""
    required_vars = [
        'MODEL_PATH',
        'S3_ENDPOINT', 
        'S3_ACCESS_KEY',
        'S3_SECRET_KEY',
        'S3_BUCKET'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables or create a .env file")
        return False
    
    print("✅ All required environment variables are set")
    return True

def check_model_file():
    """Check if model file exists"""
    model_path = os.getenv('MODEL_PATH')
    if model_path and not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    print(f"✅ Model file found: {model_path}")
    return True

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA available with {gpu_count} GPU(s)")
            for i in range(gpu_count):
                print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠️  CUDA not available, will use CPU")
            return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def main():
    parser = argparse.ArgumentParser(description="Start the ink detection inference server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    parser.add_argument("--skip-checks", action="store_true", help="Skip environment checks")
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    print("🚀 Starting Ink Detection Inference Server")
    print("=" * 50)
    
    if not args.skip_checks:
        print("Running pre-flight checks...")
        
        checks = [
            check_environment(),
            check_model_file(),
            check_gpu()
        ]
        
        if not all(checks):
            print("\n❌ Pre-flight checks failed. Use --skip-checks to bypass.")
            sys.exit(1)
        
        print("\n✅ All checks passed!")
    
    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"Workers: {args.workers}")
    print(f"Reload: {args.reload}")
    
    # Set environment variables for uvicorn
    os.environ["HOST"] = args.host
    os.environ["PORT"] = str(args.port)
    
    try:
        import uvicorn
        uvicorn.run(
            "server:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()