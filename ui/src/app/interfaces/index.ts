export interface Results {
    metadata: MetaData;
    results: GeneralizationResult[];
}

export interface DataSets {
    date: Date;
    id: string;
    label: string;
    features: string[];
}

export interface Jobs {
    date: Date;
    id: string;
    hasResults: boolean;
    metadata: MetaData;
}

export interface MetaData {
    datasetid: string;
    fits?: {
        knn: number;
        nb: number;
        svm: number;
        rf: number;
        mlp: number;
        lr: number;
        gb: number;
    };
    label?: string;
    train_negative_count?: number;
    train_positive_count?: number;
    test_negative_count?: number;
    test_positive_count?: number;
    parameters: SearchParameters;
    date: number;
}

export interface GeneralizationResult {
    key: string;
    scaler: string;
    feature_selector: string;
    algorithm: string;
    searcher: string;
    scorer: string;
    accuracy: number;
    acc_95_ci: number[];
    auc: number;
    f1: number;
    sensitivity: number;
    sn_95_ci: number[];
    specificity: number;
    sp_95_ci: number[];
    prevalence: number;
    pr_95_ci: number[];
    ppv: number;
    npv: number;
    tn: number;
    tp: number;
    fn: number;
    fp: number;
    selected_features: string;
    best_params: string;
    std_auc: number;
    mean_fpr: string;
    mean_tpr: string;
    mean_upper: string;
    mean_lower: string;
    test_fpr: string;
    test_tpr: string;
    generalization_fpr: string;
    generalization_tpr: string;
    brier_score: number;
    fop: string;
    mpv: string;
    precision_score: number;
    precision: string;
    recall: string;
}

export interface SearchParameters {
    ignore_estimator: string;
    ignore_feature_selector: string;
    ignore_scaler: string;
    ignore_scorer: string;
    ignore_searcher: string;
    ignore_shuffle: boolean;
    hyper_parameters: string;
}

export interface TaskAdded {
    id: number;
    href: string;
    pipelines: string[][];
}

export interface ActiveTaskStatus {
    current: number;
    total: number;
    status: string;
    time: number;
    id: string;
    state: 'PENDING' | 'RECEIVED' | 'STARTED' | 'REVOKED' | 'RETRY' | 'FAILURE' | 'SUCCESS';
    jobid: string;
    label: string;
    parameters: SearchParameters;
}
export interface ScheduledTaskStatus {
    eta: string;
    state: 'PENDING';
    jobid: string;
    label: string;
    parameters: SearchParameters;
}
export interface PendingTasks {
    active: ActiveTaskStatus[];
    scheduled: ScheduledTaskStatus[];
}
export interface PublishedModels {
    [key: string]: {
        date: Date;
        features: string[];
    };
}

export interface TestReply {
    predicted: number[];
    probability: number[];
    target: string;
}

export interface DataAnalysisReply {
    analysis: {
        train: DataAnalysis;
        test: DataAnalysis;
    };
    label: string;
}

export interface DataAnalysis {
    invalid: string[];
    null: {[key: string]: number};
    median: {[key: string]: number};
    mode: {[key: string]: number};
    summary: {[key: string]: DataAnalysisSummary};
    histogram: {[key: string]: [number[], number[]]};
}

export interface DataAnalysisSummary {
    '25%': number;
    '50%': number;
    '75%': number;
    count: number;
    max: number;
    min: number;
    mean: number;
    std: number;
}

export interface RefitGeneralization {
    accuracy: number;
    acc_95_ci: number[];
    avg_sn_sp: number;
    roc_auc: number;
    f1: number;
    sensitivity: number;
    sn_95_ci: number[];
    specificity: number;
    sp_95_ci: number[];
    prevalence: number;
    pr_95_ci: number[];
    ppv: number;
    npv: number;
    tn: number;
    tp: number;
    fn: number;
    fp: number;
}
