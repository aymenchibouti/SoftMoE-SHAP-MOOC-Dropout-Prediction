#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_features.py
=================
يحوّل البيانات الخام لـ KDD Cup 2015 (MOOC dropout) إلى ملف الميزات
`combined_data_processed` بأعمدته الـ576 وبنفس ترتيبها ونفس قيمها.

المدخلات المطلوبة في --data-dir :
    enrollment_train.csv   (enrollment_id, username, course_id)
    log_train.csv          (enrollment_id, time, source, event, object)
    object.csv             (course_id, module_id, category, children, start)
    truth_train.csv        (enrollment_id, label)  بدون رأس أعمدة

الاستخدام:
    python3 build_features.py --data-dir ./data --out combined_data_processed.csv.gz
    python3 build_features.py --data-dir ./data --out raw_features.csv --no-scale
    python3 build_features.py --data-dir ./data --out out.csv.gz \
            --validate combined_data_processed_csv.gz

خط الأنابيب (مستنتَج ومُتحقَّق منه مقابل الملف المرجعي):
    1) day_offset = (تاريخ الحدث − تاريخ بداية الدورة).days   ← نسبةً للدورة لا للطالب
    2) بناء 576 ميزة خام
    3) قصّ القيم الشاذة (clip) عند المئين 0.5 و 99.5 لكل عمود
    4) تقييس robust:  (x − median) / IQR   مع الرجوع إلى الانحراف المعياري
       عندما يكون IQR = 0 (وإلى 1 إذا كان الانحراف صفراً أيضاً)
    الأعمدة enrollment_id / _split / target / label / username تُترك كما هي.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ ثوابت
EVENTS = ["problem", "video", "access", "wiki", "discussion", "navigate", "page_close"]
SEQ_EVENTS = ["access", "navigate", "page_close", "video", "problem", "discussion", "wiki"]
SEQ_FL_EVENTS = ["access", "video", "problem", "discussion", "navigate", "page_close"]
MC_EVENTS = ["video", "problem", "discussion", "access"]
OBJ_CATS = ["about", "chapter", "combinedopenended", "course", "course_info",
            "dictation", "discussion", "html", "outlink", "peergrading",
            "problem", "sequential", "static_tab", "vertical", "video"]
SERVER_RAW = ["sequential", "wiki", "problem", "chapter", "navigate", "access"]
SERVER_PCT = ["chapter", "problem", "sequential"]
BROWSER_RAW = ["problem", "combinedopenended", "access", "video", "sequential"]
BROWSER_PCT = ["sequential", "combinedopenended", "video", "problem"]
# مفاتيح تُقرأ من فئة الكائن (object.csv) لا من عمود event
CATEGORY_KEYS = {"sequential", "chapter", "combinedopenended", "problem", "video"}

N_DAYS = 30
MAXLEN = 50          # طول التسلسل الأقصى في مجموعة seq_*
SESSION_GAP = 1800   # ثانية: فجوة أكبر منها تبدأ جلسة جديدة

# --- ثوابت SDA-Net (مستخرجة من create_sdanet_smart_features_csv.py) -------
# الموتّر النهائي أبعاده N × N_WEEKS × N_FEATURES × N_CHANNELS = N × 5 × 14 × 3
N_WEEKS = 5
WEEK_DAYS = N_DAYS // N_WEEKS          # 6 أيام لكل "أسبوع" لتغطية الثلاثين يوماً بخمس فترات
CHANNELS = ["MRatio", "SRatio", "Rank"]
KINDS = [("cnt", "count"), ("dur", "duration")]
# 14 ميزة = 7 أنواع أحداث × {عدد، مدة}، بترتيب EVENTS نفسه
SDANET_FEATURES = ([f"{e}_count" for e in EVENTS] + [f"{e}_duration" for e in EVENTS])
MAX_GAP_SECONDS = 1800   # سقف تقدير مدة الحدث الواحد (max_gap_seconds في سكريبت SDA-Net)
CLIP_LO, CLIP_HI = 0.5, 99.5
PASSTHROUGH = {"enrollment_id", "_split", "target", "label", "username"}

COLUMN_ORDER = [
    'enrollment_id', '_split', 'target', 'label', 'avg_chapter_delays', 'act_cnt_weekDay_01',
    'parallel_enrollments', 'act_cnt_day_01', 'act_cnt_day_02', 'act_cnt_day_03',
    'act_cnt_day_04', 'act_cnt_day_05', 'act_cnt_day_06', 'act_cnt_day_07', 'act_cnt_day_08',
    'act_cnt_day_09', 'server_sequential', 'act_cnt_day_19', 'act_cnt_day_18',
    'act_cnt_day_17', 'act_cnt_day_16', 'act_cnt_day_15', 'act_cnt_day_14', 'act_cnt_day_13',
    'act_cnt_day_12', 'act_cnt_day_11', 'act_cnt_day_10', 'act_cnt_day_29', 'wiki',
    'browser_problem', 'act_cnt_weekDay_04', 'act_cnt_weekDay_05', 'act_cnt_weekDay_06',
    'server_wiki', 'act_cnt_weekDay_00', 'act_cnt_weekDay_02', 'act_cnt_weekDay_03', 'access',
    'browser_combinedopenended', 'browser_access', 'server_chapter_percent', 'browser_video',
    'browser_sequential_percent', 'browser_combinedopenended_percent',
    'server_problem_percent', 'act_cnt_hour_22', 'class_size', 'server_sequential_percent',
    'act_cnt_hour_03', 'act_cnt_hour_02', 'act_cnt_hour_01', 'act_cnt_hour_00',
    'act_cnt_hour_07', 'act_cnt_hour_06', 'act_cnt_hour_05', 'act_cnt_hour_04',
    'act_cnt_hour_09', 'act_cnt_hour_08', 'browser_video_percent', 'navigate',
    'server_problem', 'act_cnt_hour_10', 'act_cnt_hour_11', 'act_cnt_hour_12',
    'act_cnt_hour_13', 'act_cnt_hour_14', 'act_cnt_hour_15', 'act_cnt_hour_16',
    'act_cnt_hour_17', 'act_cnt_hour_18', 'act_cnt_hour_19', 'act_cnt_day_26',
    'act_cnt_day_27', 'act_cnt_day_24', 'act_cnt_day_25', 'act_cnt_day_22', 'act_cnt_day_23',
    'act_cnt_day_20', 'act_cnt_day_21', 'server_chapter', 'act_cnt_day_28', 'act_cnt_day_30',
    'server_navigate', 'browser_problem_percent', 'browser_sequential', 'act_cnt_hour_21',
    'act_cnt_hour_20', 'act_cnt_hour_23', 'server_access', 'sessions_in_week_1',
    'sessions_in_week_0', 'sessions_in_week_3', 'sessions_in_week_2', 'sessions_in_week_4',
    'tab_total_events', 'tab_active_days', 'tab_unique_objects', 'tab_event_access',
    'tab_event_video', 'tab_event_problem', 'tab_event_discussion', 'tab_event_navigate',
    'tab_event_wiki', 'tab_event_page_close', 'tab_src_server', 'tab_src_browser',
    'tab_active_ratio', 'daily_day_00', 'daily_day_01', 'daily_day_02', 'daily_day_03',
    'daily_day_04', 'daily_day_05', 'daily_day_06', 'daily_day_07', 'daily_day_08',
    'daily_day_09', 'daily_day_10', 'daily_day_11', 'daily_day_12', 'daily_day_13',
    'daily_day_14', 'daily_day_15', 'daily_day_16', 'daily_day_17', 'daily_day_18',
    'daily_day_19', 'daily_day_20', 'daily_day_21', 'daily_day_22', 'daily_day_23',
    'daily_day_24', 'daily_day_25', 'daily_day_26', 'daily_day_27', 'daily_day_28',
    'daily_day_29', 'daily_daily_total', 'daily_daily_mean', 'daily_daily_std',
    'daily_daily_max', 'daily_daily_active_days', 'daily_daily_first7', 'daily_daily_mid14',
    'daily_daily_last7', 'daily_daily_last3', 'daily_daily_last1', 'daily_daily_zero_last7',
    'daily_daily_early_to_late_ratio', 'daily_daily_slope', 'daily_daily_recency_weighted',
    'daily_daily_entropy', 'mc_video_w1', 'mc_video_w2', 'mc_video_w3', 'mc_video_w4',
    'mc_problem_w1', 'mc_problem_w2', 'mc_problem_w3', 'mc_problem_w4', 'mc_discussion_w1',
    'mc_discussion_w2', 'mc_discussion_w3', 'mc_discussion_w4', 'mc_access_w1', 'mc_access_w2',
    'mc_access_w3', 'mc_access_w4', 'mc_video_sum', 'mc_video_mean', 'mc_video_last',
    'mc_video_trend', 'mc_video_last_ratio', 'mc_problem_sum', 'mc_problem_mean',
    'mc_problem_last', 'mc_problem_trend', 'mc_problem_last_ratio', 'mc_discussion_sum',
    'mc_discussion_mean', 'mc_discussion_last', 'mc_discussion_trend',
    'mc_discussion_last_ratio', 'mc_access_sum', 'mc_access_mean', 'mc_access_last',
    'mc_access_trend', 'mc_access_last_ratio', 'mc_all_events_w1', 'mc_all_events_w2',
    'mc_all_events_w3', 'mc_all_events_w4', 'mc_all_events_trend', 'mc_all_events_last2',
    'mc_all_events_last_ratio', 'temp_w1', 'temp_w2', 'temp_w3', 'temp_w4', 'temp_trend',
    'temp_growth_rate', 'temp_moving_avg', 'temp_volatility', 'temp_drop_indicator',
    'temp_total_activity', 'inter_event_video', 'inter_event_problem',
    'inter_event_discussion', 'inter_event_access', 'inter_active_days',
    'inter_unique_objects', 'inter_video_x_forum', 'inter_forum_per_day',
    'inter_problem_per_video', 'inter_access_per_day', 'inter_engagement_score',
    'inter_diversity_score', 'graph_degree_approx', 'graph_user_n_courses', 'graph_obj_about',
    'graph_obj_chapter', 'graph_obj_combinedopenended', 'graph_obj_course',
    'graph_obj_course_info', 'graph_obj_dictation', 'graph_obj_discussion', 'graph_obj_html',
    'graph_obj_outlink', 'graph_obj_peergrading', 'graph_obj_problem', 'graph_obj_sequential',
    'graph_obj_static_tab', 'graph_obj_vertical', 'graph_obj_video', 'seq_seq_len',
    'seq_seq_count_PAD', 'seq_seq_count_access', 'seq_seq_count_navigate',
    'seq_seq_count_page_close', 'seq_seq_count_video', 'seq_seq_count_problem',
    'seq_seq_count_discussion', 'seq_seq_count_wiki', 'seq_seq_nonpad_count',
    'seq_seq_pad_ratio', 'seq_seq_event_diversity', 'seq_seq_first10_access',
    'seq_seq_last10_access', 'seq_seq_ratio_access', 'seq_seq_first10_video',
    'seq_seq_last10_video', 'seq_seq_ratio_video', 'seq_seq_first10_problem',
    'seq_seq_last10_problem', 'seq_seq_ratio_problem', 'seq_seq_first10_discussion',
    'seq_seq_last10_discussion', 'seq_seq_ratio_discussion', 'seq_seq_first10_navigate',
    'seq_seq_last10_navigate', 'seq_seq_ratio_navigate', 'seq_seq_first10_page_close',
    'seq_seq_last10_page_close', 'seq_seq_ratio_page_close', 'seq_seq_entropy',
    'raw_inact_days_since_last_activity', 'raw_inact_last_active_day',
    'raw_inact_first_active_day', 'raw_inact_longest_zero_streak',
    'raw_inact_trailing_zero_streak', 'raw_inact_longest_active_streak',
    'raw_inact_trailing_active_streak', 'raw_inact_daily_zero_days',
    'raw_inact_daily_zero_days_ratio', 'raw_inact_daily_active_days',
    'raw_inact_daily_active_days_ratio', 'raw_inact_zero_last7', 'raw_inact_zero_last3',
    'raw_inact_active_last7', 'raw_inact_active_last3', 'raw_inact_last7_sum',
    'raw_inact_first7_sum', 'raw_inact_last3_sum', 'raw_inact_first3_sum',
    'raw_inact_last14_sum', 'raw_inact_first14_sum', 'raw_inact_last7_first7_ratio',
    'raw_inact_last3_first3_ratio', 'raw_inact_last14_first14_ratio',
    'raw_inact_last7_total_ratio', 'raw_inact_last3_total_ratio',
    'raw_inact_last_day_activity', 'raw_inact_last2_sum', 'raw_inact_last5_sum',
    'raw_inact_last10_sum', 'raw_inact_total_daily_activity', 'raw_inact_daily_mean',
    'raw_inact_daily_std', 'raw_inact_daily_cv', 'raw_inact_daily_max',
    'raw_inact_daily_slope', 'raw_inact_activity_decay_rate',
    'raw_inact_late_activity_entropy', 'raw_inact_daily_recency_weighted',
    'raw_inact_is_inactive_last_day', 'raw_inact_is_inactive_last3_all',
    'raw_inact_is_inactive_last7_all', 'raw_inact_temp_w_last_first_ratio',
    'raw_inact_temp_w_last_total_ratio', 'raw_inact_temp_w_last2_first2_ratio',
    'raw_inact_temp_w_drop_first_minus_last', 'raw_inact_temp_w_slope',
    'raw_inact_temp_w_zero_last', 'raw_inact_temp_w_zero_count',
    'raw_inact_temp_w_trailing_zero_streak', 'raw_inact_sessions_in_week_last_first_ratio',
    'raw_inact_sessions_in_week_last_total_ratio',
    'raw_inact_sessions_in_week_last2_first2_ratio',
    'raw_inact_sessions_in_week_drop_first_minus_last', 'raw_inact_sessions_in_week_slope',
    'raw_inact_sessions_in_week_zero_last', 'raw_inact_sessions_in_week_zero_count',
    'raw_inact_sessions_in_week_trailing_zero_streak',
    'raw_inact_mc_access_w_last_first_ratio', 'raw_inact_mc_access_w_last_total_ratio',
    'raw_inact_mc_access_w_last2_first2_ratio', 'raw_inact_mc_access_w_drop_first_minus_last',
    'raw_inact_mc_access_w_slope', 'raw_inact_mc_access_w_zero_last',
    'raw_inact_mc_access_w_zero_count', 'raw_inact_mc_access_w_trailing_zero_streak',
    'raw_inact_mc_problem_w_last_first_ratio', 'raw_inact_mc_problem_w_last_total_ratio',
    'raw_inact_mc_problem_w_last2_first2_ratio',
    'raw_inact_mc_problem_w_drop_first_minus_last', 'raw_inact_mc_problem_w_slope',
    'raw_inact_mc_problem_w_zero_last', 'raw_inact_mc_problem_w_zero_count',
    'raw_inact_mc_problem_w_trailing_zero_streak', 'raw_inact_mc_video_w_last_first_ratio',
    'raw_inact_mc_video_w_last_total_ratio', 'raw_inact_mc_video_w_last2_first2_ratio',
    'raw_inact_mc_video_w_drop_first_minus_last', 'raw_inact_mc_video_w_slope',
    'raw_inact_mc_video_w_zero_last', 'raw_inact_mc_video_w_zero_count',
    'raw_inact_mc_video_w_trailing_zero_streak', 'raw_inact_mc_discussion_w_last_first_ratio',
    'raw_inact_mc_discussion_w_last_total_ratio',
    'raw_inact_mc_discussion_w_last2_first2_ratio',
    'raw_inact_mc_discussion_w_drop_first_minus_last', 'raw_inact_mc_discussion_w_slope',
    'raw_inact_mc_discussion_w_zero_last', 'raw_inact_mc_discussion_w_zero_count',
    'raw_inact_mc_discussion_w_trailing_zero_streak', 'username', 'MRatio_W1_cnt_problem',
    'MRatio_W1_cnt_video', 'MRatio_W1_cnt_access', 'MRatio_W1_cnt_wiki',
    'MRatio_W1_cnt_discussion', 'MRatio_W1_cnt_navigate', 'MRatio_W1_cnt_page_close',
    'MRatio_W1_dur_problem', 'MRatio_W1_dur_video', 'MRatio_W1_dur_access',
    'MRatio_W1_dur_wiki', 'MRatio_W1_dur_discussion', 'MRatio_W1_dur_navigate',
    'MRatio_W1_dur_page_close', 'MRatio_W2_cnt_problem', 'MRatio_W2_cnt_video',
    'MRatio_W2_cnt_access', 'MRatio_W2_cnt_wiki', 'MRatio_W2_cnt_discussion',
    'MRatio_W2_cnt_navigate', 'MRatio_W2_cnt_page_close', 'MRatio_W2_dur_problem',
    'MRatio_W2_dur_video', 'MRatio_W2_dur_access', 'MRatio_W2_dur_wiki',
    'MRatio_W2_dur_discussion', 'MRatio_W2_dur_navigate', 'MRatio_W2_dur_page_close',
    'MRatio_W3_cnt_problem', 'MRatio_W3_cnt_video', 'MRatio_W3_cnt_access',
    'MRatio_W3_cnt_wiki', 'MRatio_W3_cnt_discussion', 'MRatio_W3_cnt_navigate',
    'MRatio_W3_cnt_page_close', 'MRatio_W3_dur_problem', 'MRatio_W3_dur_video',
    'MRatio_W3_dur_access', 'MRatio_W3_dur_wiki', 'MRatio_W3_dur_discussion',
    'MRatio_W3_dur_navigate', 'MRatio_W3_dur_page_close', 'MRatio_W4_cnt_problem',
    'MRatio_W4_cnt_video', 'MRatio_W4_cnt_access', 'MRatio_W4_cnt_wiki',
    'MRatio_W4_cnt_discussion', 'MRatio_W4_cnt_navigate', 'MRatio_W4_cnt_page_close',
    'MRatio_W4_dur_problem', 'MRatio_W4_dur_video', 'MRatio_W4_dur_access',
    'MRatio_W4_dur_wiki', 'MRatio_W4_dur_discussion', 'MRatio_W4_dur_navigate',
    'MRatio_W4_dur_page_close', 'MRatio_W5_cnt_problem', 'MRatio_W5_cnt_video',
    'MRatio_W5_cnt_access', 'MRatio_W5_cnt_wiki', 'MRatio_W5_cnt_discussion',
    'MRatio_W5_cnt_navigate', 'MRatio_W5_cnt_page_close', 'MRatio_W5_dur_problem',
    'MRatio_W5_dur_video', 'MRatio_W5_dur_access', 'MRatio_W5_dur_wiki',
    'MRatio_W5_dur_discussion', 'MRatio_W5_dur_navigate', 'MRatio_W5_dur_page_close',
    'SRatio_W1_cnt_problem', 'SRatio_W1_cnt_video', 'SRatio_W1_cnt_access',
    'SRatio_W1_cnt_wiki', 'SRatio_W1_cnt_discussion', 'SRatio_W1_cnt_navigate',
    'SRatio_W1_cnt_page_close', 'SRatio_W1_dur_problem', 'SRatio_W1_dur_video',
    'SRatio_W1_dur_access', 'SRatio_W1_dur_wiki', 'SRatio_W1_dur_discussion',
    'SRatio_W1_dur_navigate', 'SRatio_W1_dur_page_close', 'SRatio_W2_cnt_problem',
    'SRatio_W2_cnt_video', 'SRatio_W2_cnt_access', 'SRatio_W2_cnt_wiki',
    'SRatio_W2_cnt_discussion', 'SRatio_W2_cnt_navigate', 'SRatio_W2_cnt_page_close',
    'SRatio_W2_dur_problem', 'SRatio_W2_dur_video', 'SRatio_W2_dur_access',
    'SRatio_W2_dur_wiki', 'SRatio_W2_dur_discussion', 'SRatio_W2_dur_navigate',
    'SRatio_W2_dur_page_close', 'SRatio_W3_cnt_problem', 'SRatio_W3_cnt_video',
    'SRatio_W3_cnt_access', 'SRatio_W3_cnt_wiki', 'SRatio_W3_cnt_discussion',
    'SRatio_W3_cnt_navigate', 'SRatio_W3_cnt_page_close', 'SRatio_W3_dur_problem',
    'SRatio_W3_dur_video', 'SRatio_W3_dur_access', 'SRatio_W3_dur_wiki',
    'SRatio_W3_dur_discussion', 'SRatio_W3_dur_navigate', 'SRatio_W3_dur_page_close',
    'SRatio_W4_cnt_problem', 'SRatio_W4_cnt_video', 'SRatio_W4_cnt_access',
    'SRatio_W4_cnt_wiki', 'SRatio_W4_cnt_discussion', 'SRatio_W4_cnt_navigate',
    'SRatio_W4_cnt_page_close', 'SRatio_W4_dur_problem', 'SRatio_W4_dur_video',
    'SRatio_W4_dur_access', 'SRatio_W4_dur_wiki', 'SRatio_W4_dur_discussion',
    'SRatio_W4_dur_navigate', 'SRatio_W4_dur_page_close', 'SRatio_W5_cnt_problem',
    'SRatio_W5_cnt_video', 'SRatio_W5_cnt_access', 'SRatio_W5_cnt_wiki',
    'SRatio_W5_cnt_discussion', 'SRatio_W5_cnt_navigate', 'SRatio_W5_cnt_page_close',
    'SRatio_W5_dur_problem', 'SRatio_W5_dur_video', 'SRatio_W5_dur_access',
    'SRatio_W5_dur_wiki', 'SRatio_W5_dur_discussion', 'SRatio_W5_dur_navigate',
    'SRatio_W5_dur_page_close', 'Rank_W1_cnt_problem', 'Rank_W1_cnt_video',
    'Rank_W1_cnt_access', 'Rank_W1_cnt_wiki', 'Rank_W1_cnt_discussion', 'Rank_W1_cnt_navigate',
    'Rank_W1_cnt_page_close', 'Rank_W1_dur_problem', 'Rank_W1_dur_video', 'Rank_W1_dur_access',
    'Rank_W1_dur_wiki', 'Rank_W1_dur_discussion', 'Rank_W1_dur_navigate',
    'Rank_W1_dur_page_close', 'Rank_W2_cnt_problem', 'Rank_W2_cnt_video', 'Rank_W2_cnt_access',
    'Rank_W2_cnt_wiki', 'Rank_W2_cnt_discussion', 'Rank_W2_cnt_navigate',
    'Rank_W2_cnt_page_close', 'Rank_W2_dur_problem', 'Rank_W2_dur_video', 'Rank_W2_dur_access',
    'Rank_W2_dur_wiki', 'Rank_W2_dur_discussion', 'Rank_W2_dur_navigate',
    'Rank_W2_dur_page_close', 'Rank_W3_cnt_problem', 'Rank_W3_cnt_video', 'Rank_W3_cnt_access',
    'Rank_W3_cnt_wiki', 'Rank_W3_cnt_discussion', 'Rank_W3_cnt_navigate',
    'Rank_W3_cnt_page_close', 'Rank_W3_dur_problem', 'Rank_W3_dur_video', 'Rank_W3_dur_access',
    'Rank_W3_dur_wiki', 'Rank_W3_dur_discussion', 'Rank_W3_dur_navigate',
    'Rank_W3_dur_page_close', 'Rank_W4_cnt_problem', 'Rank_W4_cnt_video', 'Rank_W4_cnt_access',
    'Rank_W4_cnt_wiki', 'Rank_W4_cnt_discussion', 'Rank_W4_cnt_navigate',
    'Rank_W4_cnt_page_close', 'Rank_W4_dur_problem', 'Rank_W4_dur_video', 'Rank_W4_dur_access',
    'Rank_W4_dur_wiki', 'Rank_W4_dur_discussion', 'Rank_W4_dur_navigate',
    'Rank_W4_dur_page_close', 'Rank_W5_cnt_problem', 'Rank_W5_cnt_video', 'Rank_W5_cnt_access',
    'Rank_W5_cnt_wiki', 'Rank_W5_cnt_discussion', 'Rank_W5_cnt_navigate',
    'Rank_W5_cnt_page_close', 'Rank_W5_dur_problem', 'Rank_W5_dur_video', 'Rank_W5_dur_access',
    'Rank_W5_dur_wiki', 'Rank_W5_dur_discussion', 'Rank_W5_dur_navigate',
    'Rank_W5_dur_page_close', 'daily_last7', 'daily_zero_last7', 'daily_slope',
    'daily_recency_weighted', 'new_participation_density', 'peak_week_w1', 'peak_week_w2',
    'peak_week_w3', 'peak_week_w4', 'course_id_encoded'
]


# ------------------------------------------------------------------ أدوات متجهية
def batch_slope(arr):
    """ميل الانحدار الخطي لكل صف عبر أعمدة الزمن."""
    x = np.arange(arr.shape[1], dtype=float)
    xm = x.mean()
    xv = ((x - xm) ** 2).sum()
    ym = arr.mean(1)
    cov = ((arr - ym[:, None]) * (x - xm)).sum(1)
    return cov / xv


def batch_entropy(arr):
    """إنتروبيا شانون لتوزيع كل صف (صفر إذا كان المجموع صفراً)."""
    s = arr.sum(1, keepdims=True)
    p = arr / np.where(s == 0, 1, s)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(p), 0.0)
    ent = -(p * logp).sum(1)
    ent[s.ravel() == 0] = 0.0
    return ent


def longest_streak(mask):
    """أطول سلسلة متتالية من True في كل صف."""
    out = np.zeros(mask.shape[0], dtype=int)
    cur = np.zeros(mask.shape[0], dtype=int)
    for j in range(mask.shape[1]):
        cur = np.where(mask[:, j], cur + 1, 0)
        out = np.maximum(out, cur)
    return out


def trailing_streak(mask):
    """طول سلسلة True المتصلة بنهاية كل صف."""
    out = np.zeros(mask.shape[0], dtype=int)
    alive = np.ones(mask.shape[0], dtype=bool)
    for j in range(mask.shape[1] - 1, -1, -1):
        alive &= mask[:, j]
        out += alive.astype(int)
    return out


def safe_ratio(a, b):
    """a/b مع إرجاع 0 عندما يكون المقام صفراً (للنسب من نوع «حصة من إجمالي»)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(b == 0, 0.0, a / np.where(b == 0, 1.0, b))


def smooth_ratio(a, b):
    """a/(b+1) — تنعيم لابلاس، يتجنّب الانفجار عند المقام الصفري بدل تصفيره."""
    return np.asarray(a, dtype=float) / (np.asarray(b, dtype=float) + 1.0)


def pivot_count(df, col, index, categories, prefix, fmt="{}"):
    """جدول محوري لعدّ القيم مع إعادة الفهرسة على كل التسجيلات."""
    p = df.groupby(["enrollment_id", col], observed=True).size().unstack(fill_value=0)
    p = p.reindex(index=index, fill_value=0)
    out = pd.DataFrame(index=index)
    for c in categories:
        out[prefix + fmt.format(c)] = p[c].values if c in p.columns else 0
    return out


# ------------------------------------------------------------------ التحميل
def load_raw(data_dir):
    enroll = pd.read_csv(os.path.join(data_dir, "enrollment_train.csv"))
    truth = pd.read_csv(os.path.join(data_dir, "truth_train.csv"),
                        header=None, names=["enrollment_id", "label"])
    obj = pd.read_csv(os.path.join(data_dir, "object.csv"))
    obj["start"] = pd.to_datetime(obj["start"], errors="coerce")
    # object.csv فيه module_id مكرّر (499 صفاً) — إزالة التكرار ضرورية قبل الدمج
    obj_map = obj.drop_duplicates("module_id")[["module_id", "category"]]

    log = pd.read_csv(
        os.path.join(data_dir, "log_train.csv"),
        parse_dates=["time"],
        dtype={"enrollment_id": "int32", "source": "category",
               "event": "category", "object": "category"},
    )
    return enroll, truth, obj, obj_map, log


def prepare_log(log, enroll, obj_map, date_file=None):
    """إضافة day_offset (نسبةً لبداية الدورة) والساعة ويوم الأسبوع وفئة الكائن."""
    log = log.merge(enroll[["enrollment_id", "course_id"]], on="enrollment_id", how="left")

    if date_file and os.path.exists(date_file):
        dates = pd.read_csv(date_file, parse_dates=["from"])
        course_start = dates.set_index("course_id")["from"].dt.floor("D")
    else:
        course_start = log.groupby("course_id", observed=True)["time"].min().dt.floor("D")

    cs = log["course_id"].map(course_start)
    log["day_offset"] = (log["time"].dt.floor("D") - cs).dt.days.clip(0, N_DAYS - 1)
    log["hour"] = log["time"].dt.hour
    log["weekday"] = log["time"].dt.weekday
    log = log.merge(obj_map.rename(columns={"module_id": "object", "category": "obj_category"}),
                    on="object", how="left")
    return log


# ------------------------------------------------------------------ بناء الميزات
def build_features(enroll, truth, obj, obj_map, log, seq_from_start=True,
                   ratio_names="legacy"):
    idx = pd.Index(enroll["enrollment_id"].values, name="enrollment_id")
    n = len(idx)

    base = enroll.set_index("enrollment_id").reindex(idx)
    base = base.join(truth.set_index("enrollment_id"))
    base["target"] = base["label"]
    base["_split"] = "train"

    F = pd.DataFrame(index=idx)
    F["_split"] = base["_split"]
    F["target"] = base["target"]
    F["label"] = base["label"]
    F["username"] = base["username"]

    total_events = log.groupby("enrollment_id").size().reindex(idx, fill_value=0)
    total_safe = total_events.replace(0, 1).astype(float)

    # ---------------- 1) مصفوفة النشاط اليومي (أساس معظم المجموعات)
    print("  [1/11] النشاط اليومي act_cnt_day_* / daily_day_* ...")
    daily = pd.crosstab(log["enrollment_id"], log["day_offset"])
    daily = daily.reindex(index=idx, columns=range(N_DAYS), fill_value=0)
    darr = daily.values.astype(float)

    for d in range(N_DAYS):
        F[f"act_cnt_day_{d + 1:02d}"] = darr[:, d]
    for d in range(N_DAYS):
        F[f"daily_day_{d:02d}"] = darr[:, d]

    hour = pivot_count(log, "hour", idx, range(24), "act_cnt_hour_", "{:02d}")
    wday = pivot_count(log, "weekday", idx, range(7), "act_cnt_weekDay_", "{:02d}")
    F = F.join(hour).join(wday)

    # ---------------- 2) المصدر × الحدث/فئة الكائن
    print("  [2/11] أعمدة server_* / browser_* ...")
    src_evt = log.groupby(["enrollment_id", "source", "event"], observed=True).size()
    src_evt = src_evt.unstack(["source", "event"], fill_value=0).reindex(index=idx, fill_value=0)

    src_cat = log.dropna(subset=["obj_category"])
    src_cat = src_cat.groupby(["enrollment_id", "source", "obj_category"], observed=True).size()
    src_cat = src_cat.unstack(["source", "obj_category"], fill_value=0).reindex(index=idx, fill_value=0)

    src_tot_by = {sc: (src_evt[sc].sum(1).values.astype(float) if sc in src_evt.columns.levels[0]
                       else np.zeros(n)) for sc in ("server", "browser")}

    def src_col(src, key):
        # الفئات البنيوية تُؤخذ من object.csv؛ أما الأحداث الصرفة فمن عمود event
        if key in CATEGORY_KEYS and (src, key) in src_cat.columns:
            return src_cat[(src, key)].values.astype(float)
        if (src, key) in src_evt.columns:
            return src_evt[(src, key)].values.astype(float)
        if (src, key) in src_cat.columns:
            return src_cat[(src, key)].values.astype(float)
        return np.zeros(n)

    for key in SERVER_RAW:
        F[f"server_{key}"] = src_col("server", key)
    srv_safe = np.where(src_tot_by["server"] == 0, 1.0, src_tot_by["server"])
    for key in SERVER_PCT:
        F[f"server_{key}_percent"] = F[f"server_{key}"].values / srv_safe
    for key in BROWSER_RAW:
        F[f"browser_{key}"] = src_col("browser", key)
    brw_safe = np.where(src_tot_by["browser"] == 0, 1.0, src_tot_by["browser"])
    for key in BROWSER_PCT:
        F[f"browser_{key}_percent"] = F[f"browser_{key}"].values / brw_safe

    evt = pivot_count(log, "event", idx, EVENTS, "", "{}")
    for e in ["wiki", "access", "navigate"]:
        F[e] = evt[e].values

    # ---------------- 3) الجلسات
    print("  [3/11] sessions_in_week_* ...")
    log_sorted = log.sort_values(["enrollment_id", "time"], kind="mergesort")
    gap = log_sorted.groupby("enrollment_id")["time"].diff().dt.total_seconds()
    log_sorted["new_session"] = (gap.isna() | (gap > SESSION_GAP)).astype(int)
    log_sorted["session_id"] = log_sorted.groupby("enrollment_id")["new_session"].cumsum()
    log_sorted["week"] = (log_sorted["day_offset"] // 7).clip(0, 4)

    sess = (log_sorted[log_sorted["new_session"] == 1]
            .groupby(["enrollment_id", "week"]).size().unstack(fill_value=0)
            .reindex(index=idx, columns=range(5), fill_value=0))
    sarr = sess.values.astype(float)
    for w in range(5):
        F[f"sessions_in_week_{w}"] = sarr[:, w]

    # ---------------- 4) tab_*
    print("  [4/11] tab_* ...")
    active_days = (darr > 0).sum(1).astype(float)
    unique_objects = log.groupby("enrollment_id")["object"].nunique().reindex(idx, fill_value=0)

    F["tab_total_events"] = total_events.values.astype(float)
    F["tab_active_days"] = active_days
    F["tab_unique_objects"] = unique_objects.values.astype(float)
    for e in ["access", "video", "problem", "discussion", "navigate", "wiki", "page_close"]:
        F[f"tab_event_{e}"] = evt[e].values.astype(float)
    src_tot = pivot_count(log, "source", idx, ["server", "browser"], "tab_src_", "{}")
    F["tab_src_server"] = src_tot["tab_src_server"].values.astype(float)
    F["tab_src_browser"] = src_tot["tab_src_browser"].values.astype(float)
    F["tab_active_ratio"] = active_days / N_DAYS

    # ---------------- 5) إحصاءات السلسلة اليومية
    print("  [5/11] daily_daily_* ...")
    first7 = darr[:, 0:7].sum(1)
    last7 = darr[:, 23:30].sum(1)
    last3 = darr[:, 27:30].sum(1)
    weights30 = (np.arange(N_DAYS) + 1) / N_DAYS   # ترجيح خطّي متصاعد نحو نهاية الفترة

    F["daily_daily_total"] = darr.sum(1)
    F["daily_daily_mean"] = darr.mean(1)
    F["daily_daily_std"] = darr.std(1)
    F["daily_daily_max"] = darr.max(1)
    F["daily_daily_active_days"] = active_days
    F["daily_daily_first7"] = first7
    F["daily_daily_mid14"] = darr[:, 7:21].sum(1)
    F["daily_daily_last7"] = last7
    F["daily_daily_last3"] = last3
    F["daily_daily_last1"] = darr[:, 29]
    F["daily_daily_zero_last7"] = (darr[:, 23:30] == 0).sum(1).astype(float)
    F["daily_daily_early_to_late_ratio"] = smooth_ratio(first7, last7)
    F["daily_daily_slope"] = batch_slope(darr)
    F["daily_daily_recency_weighted"] = darr.dot(weights30)
    F["daily_daily_entropy"] = batch_entropy(darr)
    F["daily_last7"] = F["daily_daily_last7"].values
    F["daily_zero_last7"] = F["daily_daily_zero_last7"].values
    F["daily_slope"] = F["daily_daily_slope"].values
    F["daily_recency_weighted"] = F["daily_daily_recency_weighted"].values
    F["new_participation_density"] = active_days / N_DAYS

    # ---------------- 6) mc_* + temp_* + peak_week_*
    print("  [6/11] mc_* / temp_* / peak_week_* ...")
    log["week4"] = (log["day_offset"] // 7).clip(0, 3)
    mc_arrays = {}
    all_week = np.zeros((n, 4))
    for e in MC_EVENTS:
        sub = log[log["event"] == e]
        c = (sub.groupby(["enrollment_id", "week4"]).size().unstack(fill_value=0)
             .reindex(index=idx, columns=range(4), fill_value=0)).values.astype(float)
        mc_arrays[e] = c
        all_week += c
        for w in range(4):
            F[f"mc_{e}_w{w + 1}"] = c[:, w]
    for e in MC_EVENTS:
        c = mc_arrays[e]
        s = c.sum(1)
        F[f"mc_{e}_sum"] = s
        F[f"mc_{e}_mean"] = c.mean(1)
        F[f"mc_{e}_last"] = c[:, 3]
        F[f"mc_{e}_trend"] = c[:, 3] - c[:, 0]
        F[f"mc_{e}_last_ratio"] = safe_ratio(c[:, 3], s)
    for w in range(4):
        F[f"mc_all_events_w{w + 1}"] = all_week[:, w]
    F["mc_all_events_trend"] = all_week[:, 3] - all_week[:, 0]
    F["mc_all_events_last2"] = all_week[:, 2:4].sum(1)
    F["mc_all_events_last_ratio"] = safe_ratio(all_week[:, 3], all_week.sum(1))

    # الأسبوع الرابع يمتد 9 أيام (21..29) لتغطية الثلاثين يوماً كاملة
    weekblk = np.column_stack([darr[:, 0:7].sum(1), darr[:, 7:14].sum(1),
                               darr[:, 14:21].sum(1), darr[:, 21:30].sum(1)])
    for w in range(4):
        F[f"temp_w{w + 1}"] = weekblk[:, w]
    F["temp_trend"] = weekblk[:, 3] - weekblk[:, 0]
    F["temp_growth_rate"] = smooth_ratio(weekblk[:, 3] - weekblk[:, 0], weekblk[:, 0])
    F["temp_moving_avg"] = weekblk.mean(1)
    F["temp_volatility"] = weekblk.std(1)
    F["temp_drop_indicator"] = (np.diff(weekblk, axis=1) < 0).all(1).astype(float)
    F["temp_total_activity"] = weekblk.sum(1)

    peak = weekblk.argmax(1)
    for w in range(4):
        F[f"peak_week_w{w + 1}"] = (peak == w).astype(float)

    # ---------------- 7) inter_*
    print("  [7/11] inter_* ...")
    ad_safe = np.where(active_days == 0, 1.0, active_days)
    vid = evt["video"].values.astype(float)
    prob = evt["problem"].values.astype(float)
    disc = evt["discussion"].values.astype(float)
    acc = evt["access"].values.astype(float)

    F["inter_event_video"] = vid
    F["inter_event_problem"] = prob
    F["inter_event_discussion"] = disc
    F["inter_event_access"] = acc
    F["inter_active_days"] = active_days
    F["inter_unique_objects"] = unique_objects.values.astype(float)
    F["inter_video_x_forum"] = np.log1p(vid) * np.log1p(disc)
    F["inter_forum_per_day"] = disc / ad_safe
    F["inter_problem_per_video"] = smooth_ratio(prob, vid)
    F["inter_access_per_day"] = acc / ad_safe
    F["inter_engagement_score"] = evt[EVENTS].values.sum(1) * (active_days / N_DAYS)
    F["inter_diversity_score"] = (evt[EVENTS].values > 0).sum(1) / len(EVENTS)

    # ---------------- 8) graph_*
    print("  [8/11] graph_* ...")
    # graph_obj_* ليست تفاعلات الطالب بل حجم بنية الدورة: عدد وحدات كل فئة
    # في المقرر (تم التحقق: ارتباط 1.0000 مع الملف المرجعي).
    o_uni = obj.drop_duplicates(["course_id", "module_id"])
    cat_by_course = o_uni.groupby(["course_id", "category"]).size().unstack(fill_value=0)
    for cat in OBJ_CATS:
        col = cat_by_course[cat] if cat in cat_by_course.columns else None
        F[f"graph_obj_{cat}"] = (base["course_id"].map(col).fillna(0).values
                                 if col is not None else 0.0)
    # متوسط "درجة" العقدة في شبكة (طالب↔وحدة): أحداث الدورة ÷ عدد وحداتها
    course_events = log.groupby("course_id", observed=True).size()
    course_modules = o_uni.groupby("course_id").size()
    F["graph_degree_approx"] = (base["course_id"].map(course_events).fillna(0).values
                                / np.maximum(base["course_id"].map(course_modules).fillna(1).values, 1))
    F["graph_user_n_courses"] = base.groupby("username")["course_id"].transform("count").values

    # ---------------- 9) seq_*
    print("  [9/11] seq_* ...")
    pos = log_sorted.groupby("enrollment_id").cumcount()
    grp_n = log_sorted.groupby("enrollment_id")["time"].transform("size")
    keep = (pos < MAXLEN) if seq_from_start else (pos >= grp_n - MAXLEN)
    trunc = log_sorted[keep].copy()
    trunc["tpos"] = trunc.groupby("enrollment_id").cumcount()
    trunc["tn"] = trunc.groupby("enrollment_id")["time"].transform("size")

    seq_len = total_events.values.astype(float)                 # الطول الحقيقي غير المقصوص
    nonpad = np.minimum(seq_len, MAXLEN)
    F["seq_seq_len"] = seq_len
    seq_cnt = pivot_count(trunc, "event", idx, SEQ_EVENTS, "seq_seq_count_", "{}")
    F["seq_seq_count_PAD"] = MAXLEN - nonpad
    F = F.join(seq_cnt)
    F["seq_seq_nonpad_count"] = nonpad
    F["seq_seq_pad_ratio"] = (MAXLEN - nonpad) / MAXLEN
    F["seq_seq_event_diversity"] = (seq_cnt.values > 0).sum(1) / len(SEQ_EVENTS)

    first10 = trunc[trunc["tpos"] < 10]
    last10 = trunc[trunc["tpos"] >= trunc["tn"] - 10]
    f10 = pivot_count(first10, "event", idx, SEQ_FL_EVENTS, "f_", "{}")
    l10 = pivot_count(last10, "event", idx, SEQ_FL_EVENTS, "l_", "{}")
    for e in SEQ_FL_EVENTS:
        F[f"seq_seq_first10_{e}"] = f10[f"f_{e}"].values.astype(float)
        F[f"seq_seq_last10_{e}"] = l10[f"l_{e}"].values.astype(float)
        F[f"seq_seq_ratio_{e}"] = safe_ratio(l10[f"l_{e}"].values,
                                             seq_cnt[f"seq_seq_count_{e}"].values)
    F["seq_seq_entropy"] = batch_entropy(seq_cnt.values.astype(float))

    # ---------------- 10) raw_inact_*
    print("  [10/11] raw_inact_* ...")
    is_zero = darr == 0
    has_act = ~is_zero
    day_ix = np.arange(N_DAYS)
    last_active = np.where(has_act.any(1), N_DAYS - 1 - np.argmax(has_act[:, ::-1], 1), -1)
    first_active = np.where(has_act.any(1), np.argmax(has_act, 1), -1)

    last14, first14 = darr[:, 16:30].sum(1), darr[:, 0:14].sum(1)
    first3 = darr[:, 0:3].sum(1)
    total_daily = darr.sum(1)

    F["raw_inact_days_since_last_activity"] = np.where(last_active == -1, N_DAYS,
                                                       N_DAYS - 1 - last_active)
    F["raw_inact_last_active_day"] = last_active
    F["raw_inact_first_active_day"] = first_active
    F["raw_inact_longest_zero_streak"] = longest_streak(is_zero)
    F["raw_inact_trailing_zero_streak"] = trailing_streak(is_zero)
    F["raw_inact_longest_active_streak"] = longest_streak(has_act)
    F["raw_inact_trailing_active_streak"] = trailing_streak(has_act)
    F["raw_inact_daily_zero_days"] = is_zero.sum(1)
    F["raw_inact_daily_zero_days_ratio"] = is_zero.sum(1) / N_DAYS
    F["raw_inact_daily_active_days"] = active_days
    F["raw_inact_daily_active_days_ratio"] = active_days / N_DAYS
    F["raw_inact_zero_last7"] = is_zero[:, 23:30].sum(1)
    F["raw_inact_zero_last3"] = is_zero[:, 27:30].sum(1)
    F["raw_inact_active_last7"] = has_act[:, 23:30].sum(1)
    F["raw_inact_active_last3"] = has_act[:, 27:30].sum(1)
    F["raw_inact_last7_sum"] = last7
    F["raw_inact_first7_sum"] = first7
    F["raw_inact_last3_sum"] = last3
    F["raw_inact_first3_sum"] = first3
    F["raw_inact_last14_sum"] = last14
    F["raw_inact_first14_sum"] = first14
    F["raw_inact_last7_first7_ratio"] = smooth_ratio(last7, first7)
    F["raw_inact_last3_first3_ratio"] = smooth_ratio(last3, first3)
    F["raw_inact_last14_first14_ratio"] = smooth_ratio(last14, first14)
    F["raw_inact_last7_total_ratio"] = safe_ratio(last7, total_daily)
    F["raw_inact_last3_total_ratio"] = safe_ratio(last3, total_daily)
    F["raw_inact_last_day_activity"] = darr[:, 29]
    F["raw_inact_last2_sum"] = darr[:, 28:30].sum(1)
    F["raw_inact_last5_sum"] = darr[:, 25:30].sum(1)
    F["raw_inact_last10_sum"] = darr[:, 20:30].sum(1)
    F["raw_inact_total_daily_activity"] = total_daily
    F["raw_inact_daily_mean"] = darr.mean(1)
    F["raw_inact_daily_std"] = darr.std(1)
    F["raw_inact_daily_cv"] = safe_ratio(darr.std(1), darr.mean(1))
    F["raw_inact_daily_max"] = darr.max(1)
    F["raw_inact_daily_slope"] = batch_slope(darr)
    F["raw_inact_activity_decay_rate"] = smooth_ratio(first7 - last7, first7)
    F["raw_inact_late_activity_entropy"] = batch_entropy(darr[:, 16:30])
    F["raw_inact_daily_recency_weighted"] = darr.dot(weights30)
    F["raw_inact_is_inactive_last_day"] = (darr[:, 29] == 0).astype(float)
    F["raw_inact_is_inactive_last3_all"] = is_zero[:, 27:30].all(1).astype(float)
    F["raw_inact_is_inactive_last7_all"] = is_zero[:, 23:30].all(1).astype(float)

    def decline_block(arr, prefix):
        F[prefix + "last_first_ratio"] = smooth_ratio(arr[:, -1], arr[:, 0])
        F[prefix + "last_total_ratio"] = safe_ratio(arr[:, -1], arr.sum(1))
        F[prefix + "last2_first2_ratio"] = smooth_ratio(arr[:, -2:].sum(1), arr[:, :2].sum(1))
        F[prefix + "drop_first_minus_last"] = arr[:, 0] - arr[:, -1]
        F[prefix + "slope"] = batch_slope(arr)
        F[prefix + "zero_last"] = (arr[:, -1] == 0).astype(float)
        F[prefix + "zero_count"] = (arr == 0).sum(1)
        F[prefix + "trailing_zero_streak"] = trailing_streak(arr == 0)

    decline_block(weekblk, "raw_inact_temp_w_")
    decline_block(sarr, "raw_inact_sessions_in_week_")
    for e in ["access", "problem", "video", "discussion"]:
        decline_block(mc_arrays[e], f"raw_inact_mc_{e}_w_")

    # ---------------- 11) مستوى الدورة + MRatio/SRatio/Rank
    print("  [11/11] ميزات الدورة و MRatio_* / SRatio_* / Rank_* ...")
    chapters = obj[obj["category"] == "chapter"].dropna(subset=["start"])

    def avg_delay(s):
        v = np.sort(s.values)
        if len(v) < 2:
            return 0.0
        return float(np.mean(np.diff(v) / np.timedelta64(1, "D")))

    delays = chapters.groupby("course_id")["start"].apply(avg_delay)
    F["avg_chapter_delays"] = base["course_id"].map(delays).fillna(0.0).values
    F["class_size"] = base.groupby("course_id")["course_id"].transform("count").values.astype(float)

    # التسجيلات المتوازية: دورات أخرى لنفس المستخدم تتقاطع زمنياً مع هذه الدورة
    cstart = log.groupby("course_id", observed=True)["time"].min().dt.floor("D")
    cw = base.assign(cs=base["course_id"].map(cstart))
    cw["ce"] = cw["cs"] + pd.Timedelta(days=N_DAYS)
    par = np.zeros(n, dtype=float)
    order = {e: i for i, e in enumerate(idx)}
    for _, g in cw.groupby("username"):
        if len(g) == 1:
            continue
        s = g["cs"].values[:, None]
        e = g["ce"].values[:, None]
        ov = ((s < e.T) & (s.T < e)).sum(1) - 1
        for eid, v in zip(g.index, ov):
            par[order[eid]] = v
    F["parallel_enrollments"] = par

    codes = base["course_id"].astype("category").cat.codes.values.astype(float)
    F["course_id_encoded"] = codes / max(1, base["course_id"].nunique() - 1) - 0.5

    log["week5"] = (log["day_offset"] // WEEK_DAYS).clip(0, N_WEEKS - 1)
    ls = log_sorted
    ls["week5"] = (ls["day_offset"] // WEEK_DAYS).clip(0, N_WEEKS - 1)
    nxt = ls.groupby("enrollment_id")["time"].shift(-1)
    ls["dur"] = (nxt - ls["time"]).dt.total_seconds().clip(0, MAX_GAP_SECONDS).fillna(0)

    cnt_w = ls.groupby(["enrollment_id", "week5", "event"], observed=True).size().rename("cnt")
    dur_w = ls.groupby(["enrollment_id", "week5", "event"], observed=True)["dur"].sum().rename("dur")
    metric = pd.concat([cnt_w, dur_w], axis=1).reset_index()

    user_key = base["username"]
    course_key = base["course_id"]
    ratio_flat, tensor = build_sdanet_block(metric, idx, user_key, course_key,
                                            names=ratio_names)
    F = F.join(ratio_flat)

    F = F.reset_index()
    return F, tensor


# ------------------------------------------------------------------ SDA-Net
def build_sdanet_block(metric, idx, user_key, course_key, names="legacy"):
    """
    يبني كتلة الميزات الذكية بأسلوب SDA-Net.

    يُرجع (DataFrame مسطّح بـ210 عمود، موتّر بأبعاد N × 5 × 14 × 3).

    القنوات الثلاث لكل (أسبوع، ميزة):
      MRatio — حصة هذا التسجيل من مجموع نفس القيمة عبر كل تسجيلات المستخدم
      SRatio — حصة هذا التسجيل من مجموع نفس القيمة عبر كل تسجيلات المقرر
      Rank   — الرتبة المئينية داخل المقرر

    التسمية:
      names="legacy" → {Channel}_W{w}_{cnt|dur}_{event}   (أسماء combined_data)
      names="sdanet" → w{w}_{event}_{count|duration}_{Channel}  (تسمية SDA-Net)
    """
    n = len(idx)
    tensor = np.zeros((n, N_WEEKS, len(SDANET_FEATURES), len(CHANNELS)), dtype=np.float32)
    cols = {}
    for w in range(N_WEEKS):
        for fi, feat in enumerate(SDANET_FEATURES):
            e = feat.rsplit("_", 1)[0]
            kind_long = feat.rsplit("_", 1)[1]
            kind = "cnt" if kind_long == "count" else "dur"
            sub = metric[(metric["week5"] == w) & (metric["event"] == e)].set_index("enrollment_id")
            v = sub[kind].reindex(idx, fill_value=0).astype(float)
            ut = v.groupby(user_key.values).transform("sum")
            ct = v.groupby(course_key.values).transform("sum")
            vals = {
                "MRatio": safe_ratio(v.values, ut.values),
                "SRatio": safe_ratio(v.values, ct.values),
                "Rank": v.groupby(course_key.values).rank(pct=True).fillna(0).values,
            }
            for ci, ch in enumerate(CHANNELS):
                tensor[:, w, fi, ci] = vals[ch]
                key = (f"{ch}_W{w + 1}_{kind}_{e}" if names == "legacy"
                       else f"w{w + 1}_{feat}_{ch}")
                cols[key] = vals[ch]
    return pd.DataFrame(cols, index=idx), tensor


def save_sdanet_outputs(tensor, idx, base, csv_path, npy_path):
    """يحفظ موتّر SDA-Net الخام (.npy) وجدولاً مسطّحاً بعقد SDA-Net نفسه."""
    if npy_path:
        np.save(npy_path, tensor)
        print(f"  موتّر SDA-Net: {npy_path}  {tensor.shape}")
    if csv_path:
        cols = [f"w{w + 1}_{f}_{c}"
                for w in range(N_WEEKS) for f in SDANET_FEATURES for c in CHANNELS]
        flat = pd.DataFrame(tensor.reshape(tensor.shape[0], -1), columns=cols)
        meta = pd.DataFrame({"enrollment_id": np.asarray(idx),
                             "course_id": base["course_id"].values,
                             "dropout": base["label"].values})
        pd.concat([meta, flat], axis=1).to_csv(csv_path, index=False)
        print(f"  جدول SDA-Net: {csv_path}  ({len(flat)} × {len(cols) + 3})")


# ------------------------------------------------------------------ التقييس
def scale_features(F, clip_lo=CLIP_LO, clip_hi=CLIP_HI):
    """قصّ القيم الشاذة ثم تقييس robust: (x − median)/IQR مع بدائل عند IQR=0."""
    out = F.copy()
    params = {}
    for c in out.columns:
        if c in PASSTHROUGH:
            continue
        x = out[c].to_numpy(dtype=float)
        lo, hi = np.percentile(x, [clip_lo, clip_hi])
        x = np.clip(x, lo, hi)
        med = np.median(x)
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            s = iqr
        else:
            sd = x.std()
            s = sd if sd > 0 else 1.0
        out[c] = (x - med) / s
        params[c] = {"clip_lo": lo, "clip_hi": hi, "center": med, "scale": s}
    return out, pd.DataFrame(params).T


# ------------------------------------------------------------------ التحقق
def validate(out, ref_path):
    ref = pd.read_csv(ref_path)
    ref = ref.set_index("enrollment_id").reindex(out["enrollment_id"].values)
    got = out.set_index("enrollment_id")

    missing = [c for c in ref.columns if c not in got.columns]
    extra = [c for c in got.columns if c not in ref.columns]
    print(f"\nأعمدة مفقودة: {len(missing)} | أعمدة زائدة: {len(extra)}")
    if missing:
        print("  مفقودة:", missing[:15])
    if extra:
        print("  زائدة:", extra[:15])

    rows = []
    for c in ref.columns:
        if c not in got.columns or c in ("_split", "username"):
            continue
        a = pd.to_numeric(got[c], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(ref[c], errors="coerce").to_numpy(dtype=float)
        if np.isnan(a).all() or np.isnan(b).all():
            continue
        sa, sb = np.nanstd(a), np.nanstd(b)
        r = 1.0 if (sa == 0 and sb == 0) else (np.corrcoef(a, b)[0, 1] if sa > 0 and sb > 0 else 0.0)
        rows.append({"column": c, "corr": r,
                     "exact": float(np.mean(np.abs(a - b) < 1e-4))})
    rep = pd.DataFrame(rows)
    print(f"\nمتوسط الارتباط: {rep['corr'].mean():.5f} | "
          f"أعمدة corr>0.999: {(rep['corr'] > 0.999).sum()}/{len(rep)} | "
          f"مطابقة تامة: {(rep['exact'] > 0.999).sum()}/{len(rep)}")
    worst = rep.sort_values("corr").head(20)
    print("\nأضعف 20 عموداً:")
    print(worst.to_string(index=False))
    return rep


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="بناء combined_data_processed من البيانات الخام")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="combined_data_processed.csv.gz")
    ap.add_argument("--date-file", default=None,
                    help="date.csv اختياري (course_id, from, to) لتحديد بداية كل دورة")
    ap.add_argument("--no-scale", action="store_true", help="أخرج الميزات الخام بلا تقييس")
    ap.add_argument("--seq-from-end", action="store_true",
                    help="اقتطاع تسلسل seq_* من نهايته بدل بدايته")
    ap.add_argument("--validate", default=None, help="ملف مرجعي للمقارنة")
    ap.add_argument("--save-params", default=None, help="حفظ معاملات التقييس (CSV)")
    ap.add_argument("--ratio-names", choices=["legacy", "sdanet"], default="legacy",
                    help="تسمية كتلة الـ210: legacy = أسماء combined_data، "
                         "sdanet = w{w}_{event}_{count|duration}_{Channel}")
    ap.add_argument("--sdanet-csv", default=None,
                    help="حفظ جدول SDA-Net المسطّح (enrollment_id, course_id, dropout + 210)")
    ap.add_argument("--sdanet-npy", default=None,
                    help="حفظ موتّر SDA-Net الخام بأبعاد N×5×14×3")
    args = ap.parse_args()

    print("تحميل البيانات الخام ...")
    enroll, truth, obj, obj_map, log = load_raw(args.data_dir)
    print(f"  تسجيلات: {len(enroll):,} | أحداث: {len(log):,} | دورات: {enroll.course_id.nunique()}")

    log = prepare_log(log, enroll, obj_map, args.date_file)

    print("بناء الميزات ...")
    F, tensor = build_features(enroll, truth, obj, obj_map, log,
                               seq_from_start=not args.seq_from_end,
                               ratio_names=args.ratio_names)
    print(f"  ميزات خام: {F.shape[1]} عمود × {F.shape[0]:,} صف")

    if args.sdanet_csv or args.sdanet_npy:
        b = enroll.set_index("enrollment_id").reindex(F["enrollment_id"].values)
        b["label"] = truth.set_index("enrollment_id")["label"].reindex(b.index).values
        save_sdanet_outputs(tensor, F["enrollment_id"].values, b,
                            args.sdanet_csv, args.sdanet_npy)

    if not args.no_scale:
        print("قصّ القيم الشاذة والتقييس ...")
        F, params = scale_features(F)
        if args.save_params:
            params.to_csv(args.save_params)
            print("  حُفظت معاملات التقييس في:", args.save_params)

    if COLUMN_ORDER:
        have = [c for c in COLUMN_ORDER if c in F.columns]
        rest = [c for c in F.columns if c not in COLUMN_ORDER]
        F = F[have + rest]
        if rest:
            print(f"  تنبيه: {len(rest)} عمود خارج الترتيب المرجعي:", rest[:10])

    F.to_csv(args.out, index=False)
    print(f"تم الحفظ: {args.out}  {F.shape}")

    if args.validate:
        validate(F, args.validate)


if __name__ == "__main__":
    main()
