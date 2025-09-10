#!/usr/bin/env python3
"""
RAG System Test Runner

This script tests RAG systems to ensure they meet the requirements for:
1. Integration with the Ragent Arena for Dynamic evaluation (via /run endpoint)
2. Static evaluation support (via /evaluate endpoint)

Usage:
    python local_test.py --base-url http://localhost:5010 [options]
"""

import argparse
import sys
import json
import asyncio
import aiohttp
import time
from typing import Dict, Any, List, Optional
from pathlib import Path


class RAGSystemTester:
    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health_endpoint(self) -> bool:
        """Test if the service is running by checking /health endpoint."""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health check passed: {data}")
                    return True
                else:
                    print(f"❌ Health check failed with status {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Health check failed with error: {e}")
            return False
    
    async def test_run_endpoint(self, question: str) -> bool:
        """
        Test the /run endpoint for streaming deep research integration.
        
        Expected to work with app.py's streaming_service_producer_gen function.
        """
        print(f"\n📝 Testing /run endpoint with question: '{question}'")
        
        try:
            payload = {"question": question}
            
            async with self.session.post(
                f"{self.base_url}/run",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    print(f"❌ /run endpoint failed with status {response.status}")
                    error_body = await response.text()
                    print(f"   Error response: {error_body}")
                    return False
                
                # Check content type for streaming
                content_type = response.headers.get('content-type', '')
                if 'text/plain' not in content_type:
                    print(f"⚠️  Expected text/plain content-type, got: {content_type}")
                
                received_data = []
                intermediate_steps_received = False
                final_report_received = False
                citations_received = False
                completion_received = False
                
                # Read streaming response
                intermediate_logged = False
                final_logged = False
                citations_logged = False
                
                async for line in response.content:
                    line_text = line.decode('utf-8').strip()
                    
                    if line_text.startswith('data: '):
                        data_json = line_text[6:]  # Remove 'data: ' prefix
                        try:
                            data = json.loads(data_json)
                            received_data.append(data)
                            
                            # Check for required fields (only log once per type)
                            if 'intermediate_steps' in data and not intermediate_logged:
                                intermediate_steps_received = True
                                if data.get('is_intermediate'):
                                    print(f"   📋 Intermediate steps received")
                                    intermediate_logged = True
                            
                            if 'final_report' in data and not final_logged:
                                final_report_received = True
                                if not data.get('is_intermediate', True):
                                    print(f"   📄 Final report streaming started")
                                    final_logged = True
                            
                            if 'citations' in data and data['citations'] and not citations_logged:
                                citations_received = True
                                print(f"   🔗 Citations received: {len(data['citations'])} items")
                                citations_logged = True
                            
                            if data.get('complete'):
                                completion_received = True
                                print(f"   ✅ Completion signal received")
                        
                        except json.JSONDecodeError as e:
                            print(f"   ⚠️  Failed to parse JSON: {data_json[:100]}...")
                            continue
                
                # Validate response completeness
                success = True
                if not intermediate_steps_received:
                    print("   ❌ No intermediate_steps received")
                    success = False
                
                if not final_report_received:
                    print("   ❌ No final_report received")
                    success = False
                
                if not completion_received:
                    print("   ❌ No completion signal received")
                    success = False
                
                if success:
                    print("   ✅ /run endpoint test passed")
                    print(f"   📊 Total messages received: {len(received_data)}")
                else:
                    print("   ❌ /run endpoint test failed - missing required fields")
                
                return success
                
        except Exception as e:
            print(f"❌ /run endpoint test failed with error: {e}")
            return False
    
    async def test_evaluate_endpoint(self, validation_data: List[Dict[str, Any]]) -> bool:
        """
        Test the /evaluate endpoint for static evaluation.
        
        This test:
        1. Processes ALL validation samples through /evaluate endpoint
        2. Generates a result.jsonl file with responses
        3. Validates the file format and content
        
        Expected format from static_evaluation.md:
        Request: {"query": "string", "iid": "string"}
        Response: {"query_id": "string", "generated_response": "string"}
        """
        print(f"\n📊 Testing /evaluate endpoint - Processing {len(validation_data)} validation samples")
        print(f"   📝 Generating result.jsonl file...")
        
        results = []
        success_count = 0
        total_count = len(validation_data)
        
        # Process all validation samples
        for i, item in enumerate(validation_data):
            query = item['query']
            iid = item['iid']
            
            if i % 5 == 0 or i == total_count - 1:  # Show progress every 5 items
                print(f"   🔄 Processing sample {i+1}/{total_count}: {iid}")
            
            try:
                payload = {
                    "query": query,
                    "iid": iid  # Use iid as specified in static_evaluation.md
                }
                
                start_time = time.time()
                async with self.session.post(
                    f"{self.base_url}/evaluate",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    duration = time.time() - start_time
                    
                    if response.status != 200:
                        print(f"   ❌ Sample {i+1} failed with status {response.status}")
                        error_body = await response.text()
                        print(f"      Error: {error_body}")
                        continue
                    
                    # Parse response
                    result = await response.json()
                    
                    # Validate response format
                    if 'query_id' not in result or 'generated_response' not in result:
                        print(f"   ❌ Sample {i+1} has invalid response format")
                        continue
                    
                    # Validate query_id matches iid
                    if result['query_id'] != iid:
                        print(f"   ❌ Sample {i+1} query_id mismatch: expected {iid}, got {result['query_id']}")
                        continue
                    
                    # Add to results
                    results.append(result)
                    success_count += 1
                    
            except Exception as e:
                print(f"   ❌ Sample {i+1} failed with error: {e}")
                continue
        
        # Write results to result.jsonl file
        output_file = "result.jsonl"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
            
            print(f"   📄 Generated {output_file} with {len(results)} responses")
            
        except Exception as e:
            print(f"   ❌ Failed to write {output_file}: {e}")
            return False
        
        # Validate the generated file
        validation_success = self.validate_result_file(output_file, validation_data)
        
        # Final assessment
        success_rate = success_count / total_count
        print(f"\n   📊 Processing Results:")
        print(f"      • Total samples: {total_count}")
        print(f"      • Successful: {success_count}")
        print(f"      • Success rate: {success_rate:.1%}")
        print(f"      • Output file: {output_file}")
        
        if success_rate >= 0.8 and validation_success:  # Allow 20% failure rate
            print(f"   ✅ /evaluate endpoint test passed")
            return True
        else:
            print(f"   ❌ /evaluate endpoint test failed")
            return False
    
    def validate_result_file(self, file_path: str, validation_data: List[Dict[str, Any]]) -> bool:
        """Validate the generated result.jsonl file format and content."""
        print(f"   🔍 Validating {file_path}...")
        
        try:
            results = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            result = json.loads(line)
                            results.append(result)
                        except json.JSONDecodeError as e:
                            print(f"      ❌ Invalid JSON on line {line_num}: {e}")
                            return False
            
            # Check format compliance
            validation_iids = {item['iid'] for item in validation_data}
            result_iids = set()
            
            for i, result in enumerate(results):
                # Check required fields
                if 'query_id' not in result:
                    print(f"      ❌ Missing 'query_id' in result {i+1}")
                    return False
                
                if 'generated_response' not in result:
                    print(f"      ❌ Missing 'generated_response' in result {i+1}")
                    return False
                
                query_id = result['query_id']
                generated_response = result['generated_response']
                
                # Check query_id is from validation set
                if query_id not in validation_iids:
                    print(f"      ❌ Unknown query_id: {query_id}")
                    return False
                
                result_iids.add(query_id)
                
                # Check response is not empty
                if not generated_response or not generated_response.strip():
                    print(f"      ❌ Empty response for query_id: {query_id}")
                    return False
            
            # Check completeness
            missing_iids = validation_iids - result_iids
            if missing_iids:
                print(f"      ⚠️  Missing responses for {len(missing_iids)} samples: {list(missing_iids)[:5]}...")
            
            duplicate_iids = len(results) - len(result_iids)
            if duplicate_iids > 0:
                print(f"      ❌ Found {duplicate_iids} duplicate query_ids")
                return False
            
            print(f"      ✅ File format validation passed")
            print(f"      📋 Contains {len(results)} valid responses")
            return True
            
        except Exception as e:
            print(f"      ❌ File validation failed: {e}")
            return False


def load_validation_data(file_path: str) -> List[Dict[str, Any]]:
    """Load validation data from JSONL file."""
    validation_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    validation_data.append(data)
        
        print(f"📂 Loaded {len(validation_data)} validation samples from {file_path}")
        return validation_data
    
    except FileNotFoundError:
        print(f"❌ Validation file not found: {file_path}")
        return []
    except Exception as e:
        print(f"❌ Error loading validation data: {e}")
        return []


async def main():
    parser = argparse.ArgumentParser(
        description='Test RAG system endpoints for compliance with DeepResearch Comparator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full test (both endpoints) - Default mode
    python local_test.py --base-url http://localhost:5010
    
    # Test only /evaluate endpoint
    python local_test.py --base-url http://localhost:5010 --test-mode evaluate
    
    # Test only /run endpoint
    python local_test.py --base-url http://localhost:5010 --test-mode run
    
    # Full test with custom settings
    python local_test.py --base-url http://localhost:5010 --test-mode full \\
        --validation-file custom_val.jsonl \\
        --test-question "What is machine learning?"
        """
    )
    
    parser.add_argument(
        '--base-url', 
        type=str, 
        required=True,
        help='Base URL of the RAG service (e.g., http://localhost:5010)'
    )
    
    parser.add_argument(
        '--test-mode',
        type=str,
        choices=['full', 'evaluate', 'run'],
        default='full',
        help='Test mode: full (both endpoints), evaluate (only /evaluate), run (only /run). Default: full'
    )
    
    parser.add_argument(
        '--validation-file',
        type=str,
        default='./t2t_val.jsonl',
        help='Path to validation JSONL file (default: ./t2t_val.jsonl)'
    )
    
    parser.add_argument(
        '--test-question',
        type=str,
        default='What are the benefits of renewable energy?',
        help='Question to test the /run endpoint'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Request timeout in seconds (default: 300)'
    )
    
    args = parser.parse_args()
    
    print("🧪 RAG System Compliance Tester")
    print("=" * 50)
    print(f"Target URL: {args.base_url}")
    print(f"Test Mode: {args.test_mode}")
    if args.test_mode in ['full', 'evaluate']:
        print(f"Validation file: {args.validation_file}")
    if args.test_mode in ['full', 'run']:
        print(f"Test question: {args.test_question}")
    print()
    
    # Load validation data only if needed
    validation_data = []
    if args.test_mode in ['full', 'evaluate']:
        validation_data = load_validation_data(args.validation_file)
        if not validation_data:
            print("⚠️  No validation data loaded. /evaluate test will be skipped.")
    
    # Run tests
    all_tests_passed = True
    
    async with RAGSystemTester(args.base_url, args.timeout) as tester:
        # Health check
        health_ok = await tester.test_health_endpoint()
        if not health_ok:
            print("\n❌ Health check failed. Cannot proceed with other tests.")
            sys.exit(1)
        
        # Test /run endpoint
        if args.test_mode in ['full', 'run']:
            run_ok = await tester.test_run_endpoint(args.test_question)
            all_tests_passed = all_tests_passed and run_ok
        else:
            print("\n⏭️  Skipping /run endpoint test (not selected in test mode)")
        
        # Test /evaluate endpoint
        if args.test_mode in ['full', 'evaluate'] and validation_data:
            evaluate_ok = await tester.test_evaluate_endpoint(validation_data)
            all_tests_passed = all_tests_passed and evaluate_ok
        elif args.test_mode in ['full', 'evaluate']:
            print("\n⏭️  Skipping /evaluate endpoint test (no validation data)")
        else:
            print("\n⏭️  Skipping /evaluate endpoint test (not selected in test mode)")
    
    # Final results
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! This RAG system is compliant.")
        print("✅ Ready for integration with DeepResearch Comparator")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("💡 Check the endpoint implementations and try again.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())