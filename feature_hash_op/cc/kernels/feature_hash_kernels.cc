/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op_kernel.h"
#include "MurmurHash2.h"

using namespace tensorflow;


class FeatureHasher {
public:
    const uint32_t m_MaxProbeCount = 10240 * 4;
    const static size_t HashTableFull = -1;
    const static size_t HashTableNotFound = -2;
    const uint8_t HashTableOccupyFlag = 1;

    FeatureHasher(char* ptr, size_t length, size_t size_per_bucket, size_t size_per_key) : m_hashTable(ptr), m_length(length), m_sizePerBucket(size_per_bucket), m_sizePerKey(size_per_key) {
        LOG(INFO) << "length: " << length << " \n";
        LOG(INFO) << "size_per_bucket: " << size_per_bucket << " \n";

        m_bucketsize = length / size_per_bucket;
        LOG(INFO) << "initialize m_bucketsize: " << m_bucketsize << " \n";

    }

    bool InsertKey(const char* raw_key, int64_t& index, uint32_t& probe_count) {

        memcpy(m_hashTable + index * m_sizePerBucket, raw_key, m_sizePerKey);

        memcpy(const_cast<char*>(m_hashTable + index * m_sizePerBucket + m_sizePerKey), &HashTableOccupyFlag, 1);
        return true;
    }

    inline int64_t GetKeyHash(const void* key)
    {
        uint64_t keyHash = MurmurHash64A(key, m_sizePerKey, m_murmurHashSeed);

        auto h =  static_cast<int64_t>(keyHash & 0x7fffffffffffffff);
        return h;
    }

    // return the index of key if the key exists in the hash table, or insert the key in the hash table and return the index
    size_t GetOrInsert(const char* raw_key) {
        int64_t index = GetKeyHash(raw_key) % m_bucketsize;

        uint32_t probe_count = 0;
        bool found = false;

        const auto max_probe_count = m_MaxProbeCount > m_bucketsize ? m_bucketsize : m_MaxProbeCount;
        //LOG(INFO) << "m_bucketsize: " << m_bucketsize << " \n";

        while (probe_count < max_probe_count) {
            uint8_t* bucketFlag = reinterpret_cast<uint8_t*>(m_hashTable + index * m_sizePerBucket + m_sizePerKey);
            if (*bucketFlag == HashTableOccupyFlag)
            {
                if (memcmp(m_hashTable + index * m_sizePerBucket, raw_key, m_sizePerKey) == 0) {
                    found = true;
                    break;
                }
                else {
                    index = (index + 1) % m_bucketsize;
                    probe_count++;
                }
            }
            
            else {
            // key is not found

                found = InsertKey(raw_key, index, probe_count);
                if (found) {
                    break;
                }

            }
    
        }

        if (probe_count >= max_probe_count) {
            return HashTableFull;
        }

        if (found) {
            return index;
        }

        return HashTableNotFound;
  }

  private:
      const uint64_t m_murmurHashSeed = 0xfdecdcab36ab;
      char* m_hashTable;
      size_t m_length;
      size_t m_sizePerKey;
      size_t m_sizePerBucket;
      size_t m_bucketsize;
};

class FeatureHashOp : public OpKernel {
 public:
  explicit FeatureHashOp(OpKernelConstruction* context) : OpKernel(context) {
    context->GetAttr("size_per_bucket", &m_sizePerBucket);
    context->GetAttr("size_per_key", &m_sizePerKey);
  }

  void Compute(OpKernelContext* context) override {
    Tensor featureHashTable = context->mutable_input(0, false);
    OP_REQUIRES(context, featureHashTable.IsInitialized(),
                errors::InvalidArgument("FeatureHashTable is not initialized. "));
    //Tensor&  featureHashTable = context->input(0);
    auto featureHashTableVec = featureHashTable.flat<int32>();

    const Tensor& keys = context->input(1);
    auto keysVec = keys.flat<int32>();

    const int64_t bucketSize = sizeof(int32) * featureHashTableVec.size() / m_sizePerBucket;
    const uint32_t numKeys = static_cast<uint32_t>(sizeof(int32)*keysVec.size()/m_sizePerBucket);

    Tensor* featureIds = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({numKeys}), &featureIds));
    auto featureIdsVec = featureIds->vec<int64>();

    FeatureHasher featureHasher((char*)featureHashTable.tensor_data().data(), featureHashTable.tensor_data().size(), m_sizePerBucket, m_sizePerKey);

    //LOG(INFO) << "Num of Keys: " << numKeys << " \n";

    for(int keyId = 0; keyId < numKeys; keyId++)
    {
      int64 featureId = 0;

      featureId = featureHasher.GetOrInsert((char*)(keys.tensor_data().data()) + keyId * m_sizePerBucket);

      if(featureId == FeatureHasher::HashTableFull)
      {
        OP_REQUIRES(
            context,
            false,
            errors::Internal(
                "failed to get or insert the key"
                 ));
      }
      featureIdsVec(keyId) = featureId;
      //LOG(INFO) << "Process " << keyId << " feature\n";
    }
  }

private:
  int m_sizePerBucket;
  int m_sizePerKey;
};

REGISTER_KERNEL_BUILDER(Name("FeatureHashOp").Device(DEVICE_CPU), FeatureHashOp);

