--"INTEL CONFIDENTIAL"
--Copyright 2017 Intel Corporation All Rights Reserved.
--
--The source code contained or described herein and all documents related to the source code ("Material") are owned by Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
--
--No license under any patent, copyright, trade secret or other intellectual property right is granted to or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement, estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel in writing.

-- ###########################
-- parallel order by. required by queries:
-- Note the "bigbench." prefix! Actual enabling is query statement specific and
-- only activated where required to achive a deterministic output.
-- ###########################
set bigbench.hive.optimize.sampling.orderby=true;
set bigbench.hive.optimize.sampling.orderby.number=20000;
set bigbench.hive.optimize.sampling.orderby.percent=0.1;

--  allow implicit cross joins (cartesian products)
set spark.sql.crossJoin.enabled = true;

-- Database - DO NOT DELETE OR CHANGE
CREATE DATABASE IF NOT EXISTS ${env:BIG_BENCH_DATABASE};
use ${env:BIG_BENCH_DATABASE};


