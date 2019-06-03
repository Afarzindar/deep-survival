from FeatureEncoder import CategoricalEncoder, DataFrameSelector

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn_pandas import CategoricalImputer

import pandas as pd
import numpy as np
import tensorflow as tf

import math
import sys


HIDDEN_UNITS = [256, 128, 64]
TIME_UNITS = 24
DROPOUT_PROB_TRAIN = [0.5, 0.5, 0.5]
DROPOUT_PROB_TEST = [1.0, 1.0, 1.0]

tf.set_random_seed(7)
np.random.seed(7)

def processData():

    train = pd.read_csv("train.csv")

    catFeatures = []
    numFeatures = []
    for name, val in zip(train.columns, train.dtypes):
        if val in [np.dtype('O'), np.dtype('int64')]:
            if name not in ['GTIME', 'GSTATUS_THREE_MONTHS', 'GSTATUS_SIX_MONTHS', 'GSTATUS_ONE_YEAR', 'GSTATUS_THREE_YEARS']:
                catFeatures.append(name)
        else:
            numFeatures.append(name)

    # catFeatures = ['GENDER', 'ABO', 'LIFE_SUP_TCR', 'MALIG_TCR', 'EXC_HCC', 'EXC_CASE', 'PERM_STATE', 'PREV_AB_SURG_TCR', 'BACT_PERIT_TCR', 'PORTAL_VEIN_TCR', 'TIPSS_TCR', 'WORK_INCOME_TCR', 'INIT_DIALYSIS_PRIOR_WEEK', 'INIT_MELD_OR_PELD', 'FINAL_DIALYSIS_PRIOR_WEEK', 'FINAL_MELD_OR_PELD', 'PERM_STATE_TRR', 'WORK_INCOME_TRR', 'MALIG_TRR', 'LIFE_SUP_TRR', 'PORTAL_VEIN_TRR', 'PREV_AB_SURG_TRR', 'TIPSS_TRR', 'HBV_CORE', 'HBV_SUR_ANTIGEN', 'HCV_SEROSTATUS', 'EBV_SEROSTATUS', 'HIV_SEROSTATUS', 'CMV_STATUS', 'CMV_IGG', 'CMV_IGM', 'TXLIV', 'PREV_TX', 'DDAVP_DON', 'CMV_DON', 'HEP_C_ANTI_DON', 'HBV_CORE_DON', 'HBV_SUR_ANTIGEN_DON', 'DON_TY', 'GENDER_DON', 'HOME_STATE_DON', 'NON_HRT_DON', 'ANTIHYPE_DON', 'PT_DIURETICS_DON', 'PT_STEROIDS_DON', 'PT_T3_DON', 'PT_T4_DON', 'VASODIL_DON', 'VDRL_DON', 'CLIN_INFECT_DON', 'EXTRACRANIAL_CANCER_DON', 'HIST_CIG_DON', 'HIST_COCAINE_DON', 'DIABETES_DON', 'HIST_HYPERTENS_DON', 'HIST_OTH_DRUG_DON', 'ABO_DON', 'INTRACRANIAL_CANCER_DON', 'SKIN_CANCER_DON', 'HIST_CANCER_DON', 'PT_OTH_DON', 'HEPARIN_DON', 'ARGININE_DON', 'INSULIN_DON', 'DIAL_TX', 'ABO_MAT', 'AGE_GROUP', 'MALIG', 'RECOV_OUT_US', 'TATTOOS', 'LI_BIOPSY', 'PROTEIN_URINE', 'CARDARREST_NEURO', 'INOTROP_SUPPORT_DON', 'CDC_RISK_HIV_DON', 'HISTORY_MI_DON', 'CORONARY_ANGIO_DON', 'LT_ONE_WEEK_DON']
    # numFeatures = ['WGT_KG_DON_CALC', 'INIT_INR', 'ETHCAT_DON', 'ETHNICITY', 'DGN_TCR', 'REM_CD', 'INIT_AGE', 'ALBUMIN_TX', 'BMI_DON_CALC', 'EXC_EVER', 'OTH_LIFE_SUP_TCR', 'FINAL_ASCITES', 'WGT_KG_CALC', 'END_BMI_CALC', 'LISTYR', 'DDR1', 'FINAL_ALBUMIN', 'DB2', 'INIT_BMI_CALC', 'CITIZENSHIP', 'DB1', 'EDUCATION', 'DAYSWAIT_CHRON', 'OTH_LIFE_SUP_TRR', 'MED_COND_TRR', 'INIT_WGT_KG', 'MELD_PELD_LAB_SCORE', 'NUM_PREV_TX', 'INIT_SERUM_SODIUM', 'VENTILATOR_TCR', 'TX_PROCEDUR_TY', 'LITYP', 'INIT_SERUM_CREAT', 'WGT_KG_TCR', 'TBILI_DON', 'HGT_CM_CALC', 'SGOT_DON', 'ASCITES_TX', 'INIT_MELD_PELD_LAB_SCORE', 'ECD_DONOR', 'CREAT_TX', 'INIT_ENCEPH', 'INIT_HGT_CM', 'PRI_PAYMENT_TRR', 'INIT_STAT', 'ARTIFICIAL_LI_TCR', 'PT_CODE', 'WL_ID_CODE', 'INIT_ALBUMIN', 'ARTIFICIAL_LI_TRR', 'AGE_DON', 'ON_VENT_TRR', 'PRI_PAYMENT_TCR', 'BLOOD_INF_DON', 'CREAT_DON', 'REGION', 'INIT_ASCITES', 'HEMATOCRIT_DON', 'DIAB', 'TBILI_TX', 'FINAL_INR', 'AGE', 'FUNC_STAT_TRR', 'ETHCAT', 'CITIZENSHIP_DON', 'DEATH_MECH_DON', 'FUNC_STAT_TCR', 'FINAL_SERUM_SODIUM', 'COD_CAD_DON', 'FINAL_BILIRUBIN', 'BUN_DON', 'END_STAT', 'BMI_CALC', 'DDR2', 'FINAL_SERUM_CREAT', 'HIST_DIABETES_DON', 'ENCEPH_TX', 'SHARE_TY', 'DA1', 'PH_DON', 'FINAL_MELD_PELD_LAB_SCORE', 'BMI_TCR', 'INIT_BILIRUBIN', 'DISTANCE', 'SGPT_DON', 'PULM_INF_DON', 'HGT_CM_TCR', 'TRANSFUS_TERM_DON', 'FINAL_ENCEPH', 'DIAG', 'DA2', 'HGT_CM_DON_CALC', 'URINE_INF_DON', 'COLD_ISCH', 'INR_TX', 'DEATH_CIRCUM_DON', 'CANCER_SITE_DON']

    #Categorical pipeline
    cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(catFeatures)),
            ('imputer', CategoricalImputer()),
            ('cat_encoder', CategoricalEncoder("onehot-dense", handle_unknown='ignore')),      
        
    ])
        
    #Numerical pipeline
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(numFeatures)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    #Full pipeline
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),

    ])

    # train = pd.read_csv("train.csv")
    X_train = full_pipeline.fit_transform(train.loc[:, catFeatures + numFeatures])
    gstatusSixMonths_train = train["GSTATUS_SIX_MONTHS"].values
    gstatusOneYear_train = train["GSTATUS_ONE_YEAR"].values
    gstatusThreeYears_train = train["GSTATUS_THREE_YEARS"].values
    gstatus_train = train["GSTATUS_THREE_YEARS"].values
    gtime_train = train["GTIME"].values
    Y_train = np.array([[gstatus_train[i], gtime_train[i]] for i in range(len(gtime_train))]) #[is_not_censored, survival time]

    test = pd.read_csv("test.csv")
    X_test = full_pipeline.transform(test.loc[:, catFeatures + numFeatures])
    gstatusSixMonths_test = test["GSTATUS_SIX_MONTHS"].values
    gstatusOneYear_test = test["GSTATUS_ONE_YEAR"].values
    gstatusThreeYears_test = test["GSTATUS_THREE_YEARS"].values
    gstatus_test = test["GSTATUS_THREE_YEARS"].values
    gtime_test = test["GTIME"].values
    Y_test = np.array([[gstatus_test[i], gtime_test[i]] for i in range(len(gtime_test))]) #[is_not_censored, survival time]

    return X_train, Y_train, X_test, Y_test


def model(X, hidden_units, num_time_units, dropout_keep_prob, batch_norm_epsilon=1e-7, train=True, reuse=False):

    fc_layers = []
    num_layers = len(hidden_units)

    with tf.variable_scope("DL", reuse=reuse):

        prev_layer = X

        for i in range(num_layers):
            layer_out = tf.layers.dense(prev_layer, hidden_units[i], kernel_initializer=tf.contrib.layers.xavier_initializer())
            norm_out = tf.layers.batch_normalization(layer_out, epsilon=batch_norm_epsilon, training = train, center=False, scale=False)
            activation_out = tf.nn.relu(norm_out)

            dropout_out = tf.nn.dropout(activation_out, dropout_keep_prob[i])

            prev_layer = dropout_out

        output_score1 = tf.layers.dense(prev_layer, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)

        logits = tf.layers.dense(output_score1, num_time_units, kernel_initializer=tf.contrib.layers.xavier_initializer())
        output_score2 = tf.nn.sigmoid(logits)

        return output_score1, output_score2



def py_func(func, inp, Tout, stateful=True, name=None, grad_func=None):

    grad_name = 'PyFuncGrad_' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(grad_name)(grad_func)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": grad_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)



def unique_set(Y_hazard):

    a = Y_hazard

    # Get unique times
    t, idx = np.unique(a, return_inverse=True)

    # Get indexes of sorted array
    sort_idx = np.argsort(a)

    # Sort the array using the index
    a_sorted = a[sort_idx]

    # Find duplicates and make them 0
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))

    # Difference a[n+1] - a[n] of non zero indexes (Gives index ranges of patients with same timesteps)
    unq_count = np.diff(np.nonzero(unq_first)[0])

    # Split all index from single array to multiple arrays where each contains all indexes having same timestep
    unq_idx = np.split(sort_idx, np.cumsum(unq_count))

    return t, unq_idx


def cox_partial_likehood_loss(score, Y_hazard, Y_survival, epsilon=1e-8):

    t, H = unique_set(Y_hazard)

    actual_event_index = np.nonzero(Y_survival)[0]
    H = [list(set(h) & set(actual_event_index)) for h in H]
    n = [len(h) for h in H]

    total = 0.0
    for j in range(len(t)):
        total_1 = np.sum(np.log(np.absolute(score[H[j]] + epsilon)))
        m = n[j]
        total_2 = 0.0
        for i in range(m):
            subtotal = np.sum(score[sum(H[j:],[])]) - (i * 1.0 / m) * (np.sum(score[H[j]]))
            subtotal = np.log(np.absolute(subtotal + epsilon))
            total_2 = total_2 + subtotal
        total = total + total_1 - total_2
    return np.float32(total)


def cox_partial_likehood_loss_grad_impl(score, Y_hazard, Y_survival, grad, epsilon=1e-8):

    t, H = unique_set(Y_hazard)
    keep_index = np.nonzero(Y_survival)[0]  #censor = 1
    H = [list(set(h)&set(keep_index)) for h in H]
    n = [len(h) for h in H]
    
    total = np.zeros(score.shape)
    for j in range(len(t)):
        tmp = 1.0 / (score + epsilon)
        mask = np.ones(tmp.shape, dtype=bool)
        mask[H[j]] = False
        tmp[mask] = 0.0
        total_1 = tmp
        m = n[j]
        total_2 = np.zeros(score.shape, dtype=np.float32)
        for i in range(m):
            tmp1 = np.zeros(score.shape, dtype=np.float32)              
            tmp2 = np.zeros(score.shape, dtype=np.float32)
            tmp1[sum(H[j:],[])] = 1
            tmp2[H[j]] = (i*1.0/m)
            num = tmp1 - tmp2
            denom = np.sum(score[sum(H[j:],[])]) - (i*1.0/m) * (np.sum(score[H[j]])) + epsilon
            subtotal = num / denom
            total_2 += subtotal
        total = total + total_1 - total_2
    return np.float32(total * grad)



def cox_partial_likehood_loss_grad(op, grad):

    score = op.inputs[0]
    Y_hazard = op.inputs[1]
    Y_survival = op.inputs[2]
    
    return tf.py_func(cox_partial_likehood_loss_grad_impl, [score, Y_hazard, Y_survival, grad], grad.dtype), tf.zeros(tf.shape(Y_hazard)), tf.zeros(tf.shape(Y_survival))


def acc_pairs(Y_hazard, Y_survival):

    tmp1 = Y_hazard <= 30 * TIME_UNITS
    tmp1 = tmp1.astype(int)
    tmp2 = Y_survival.astype(int)
    noncensor = (tmp2) & (tmp1)
    noncensor_index = np.nonzero(noncensor)[0]
    acc_pair = []
    for i in noncensor_index:
        all_j =  np.array(range(len(Y_hazard)))[Y_hazard > Y_hazard[i]]
        acc_pair += [(i,j) for j in all_j]

    return acc_pair


def isotonic_rank_loss(score, Y_hazard, Y_survival, num_time_units, score_prev_layer):

    acc_pair = acc_pairs(Y_hazard, Y_survival)
    total = []

    for i, j in acc_pair:
        yj = np.ones((num_time_units, 1))
        yi = np.ones((num_time_units, 1))

        yi[int(np.ceil(Y_hazard[i] / 30)):, 0] = 0
        yj[int(np.ceil(Y_hazard[j] / 30)):, 0] = 0

        rank_loss = yj * (1 - yi)

        regression_loss = np.multiply(score[j] - score[i] - 1, rank_loss)
        regression_loss = np.linalg.norm(regression_loss)

        total.append(regression_loss)

    return np.float32(np.sum(total))



def isotonic_rank_loss_grad_impl(score, Y_hazard, Y_survival, num_time_units, score_prev_layer, grad):

    acc_pair = acc_pairs(Y_hazard, Y_survival)    
    total = 0.0
    for i,j in acc_pair: #Change to j,i instead of i,j in original code

        yj = np.ones((num_time_units, 1))
        yi = np.ones((num_time_units, 1))

        yi[int(np.ceil(Y_hazard[i] / 30)):, 0] = 0
        yj[int(np.ceil(Y_hazard[j] / 30)):, 0] = 0

        rank_loss = yj * (1 - yi)
        # print score[j] - score[i] - 1

        score_mask = np.multiply(score[j] - score[i] - 1, rank_loss)
        L2dist = np.linalg.norm(score_mask)

        sig1 = score[j] * (1 - score[j])
        sig2 = score[i] * (1 - score[i])

        x_mask = np.multiply(rank_loss, (sig1 * score_prev_layer[j] - sig2 * score_prev_layer[i]))
        
        diff_value = np.dot(score_mask.T, x_mask)
        total = total + diff_value / (L2dist + 1e-8)

    return np.float32(total * grad)


def isotonic_rank_loss_grad(op, grad):

    score = op.inputs[0]
    Y_hazard = op.inputs[1]
    Y_survival = op.inputs[2]
    num_time_units = op.inputs[3]
    score_prev_layer = op.inputs[4]
    return tf.py_func(isotonic_rank_loss_grad_impl, [score, Y_hazard, Y_survival, num_time_units, score_prev_layer, grad], grad.dtype), tf.zeros(tf.shape(Y_hazard)), tf.zeros(tf.shape(Y_survival)), tf.zeros(1)[0], tf.zeros(tf.shape(score_prev_layer))


def CIndex(Y_hazard, Y_survival, score):

    acc_pair = acc_pairs(Y_hazard, Y_survival)
    prob = np.sum([score[i] <= score[j] for i, j in acc_pair]) * 1.0 / len(acc_pair)
    return prob


def train(sess, X_train, Y_hazard_train, Y_survival_train, BATCH_SIZE=32, learning_rate=0.001, epochs=100, display_rate=1):

    M, inp_size = X_train.shape
    num_batches = int(math.ceil(float(M) / BATCH_SIZE))

    X = tf.placeholder(tf.float32, shape=[None, inp_size], name="X")
    Y_hazard = tf.placeholder(tf.float32, shape=[None,], name="Y_hazard")
    Y_survival = tf.placeholder(tf.float32, shape=[None,], name="Y_survival")

    score1, score2 = model(X, HIDDEN_UNITS, TIME_UNITS, DROPOUT_PROB_TRAIN)

    loss1 = py_func(cox_partial_likehood_loss,
        [score1, Y_hazard, Y_survival], [tf.float32],
        name = "cox_partial_likehood_loss",
        grad_func = cox_partial_likehood_loss_grad)[0]

    loss2 = py_func(isotonic_rank_loss,
        [score2, Y_hazard, Y_survival, TIME_UNITS, score1], [tf.float32],
        name = "isotonic_rank_loss",
        grad_func = isotonic_rank_loss_grad)[0]

    # # Check how to combine the two losses
    joint_loss = loss1 + loss2
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(joint_loss)
    # optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss2)

    cost = []

    # sess = tf.Session()

    init = tf.global_variables_initializer()
    sess.run(init)

 
    for e in range(epochs):
        epoch_loss = 0.0
        for i in range(num_batches):

            if i == num_batches - 1:
                x = X_train[i * BATCH_SIZE:, :]
                y_h = Y_hazard_train[i * BATCH_SIZE:]
                y_s = Y_survival_train[i * BATCH_SIZE:]

            else:
                x = X_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE, :]
                y_h = Y_hazard_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                y_s = Y_survival_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]

            sort_idx = np.argsort(-y_h)
            x = x[sort_idx, :]
            y_h = y_h[sort_idx]
            y_s = y_s[sort_idx]

            _, loss = sess.run([optimizer, joint_loss], feed_dict = {X: x, Y_hazard: y_h, Y_survival: y_s})

            epoch_loss = (loss / BATCH_SIZE)

        if i % display_rate == 0:
            print "Epoch " + str(e) + ", loss: " + str(epoch_loss)
            cost.append((e, epoch_loss))

    return sess


def test(X_test, Y_hazard_test, Y_survival_test, _train=True):

    M, inp_size = X_test.shape
    X = tf.placeholder(tf.float32, shape=[None, inp_size], name="X")

    score1, score2 = model(X, HIDDEN_UNITS, TIME_UNITS, DROPOUT_PROB_TEST, batch_norm_epsilon=1e-7, train=False, reuse=_train)

    return X, score1, score2


def evaluate(Y_hazard, Y_survival, score):

    return CIndex(Y_hazard, Y_survival, score)


def main():

    X_train, Y_train, X_test, Y_test = processData()

    print "X_train: ", X_train.shape
    print "Y_train: ", Y_train.shape
    print "X_test: ", X_test.shape
    print "Y_test: ", Y_test.shape

    # Add input normalization later

    BATCH_SIZE = 32

    sess = tf.Session()

    _train = int(sys.argv[1])

    if _train == 1:
        sess = train(sess, X_train, Y_train[:, 1], Y_train[:, 0])
        saver = tf.train.Saver()
        saver.save(sess, "model/model.ckpt")

    if _train == 1:
        X, score1, score2 = test(X_test, Y_test[:, 1], Y_test[:, 0], _train=True)
    else:
        X, score1, score2 = test(X_test, Y_test[:, 1], Y_test[:, 0], _train=False)

    if _train == 0:
        saver = tf.train.Saver()
        saver.restore(sess, "model/model.ckpt")

    score1, score2 = sess.run([score1, score2], feed_dict = {X: X_test})

    print evaluate(Y_test[:, 1], Y_test[:, 0], score1)

if __name__ == "__main__":
    main()
