// Chart.js
import Chart from 'chart.js'
// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

$(document).ready(function () {
    let _labels;
    let _data;

    $.ajax({
        dataType: "json",
        url: $AJAX_TEXTS_LABELS_URL,
        success: function (data) {
            _labels = data.labels;
            _data = data.data;
        },
        complete: function () {
            const ctx = document.getElementById("textsBarChart");
            new Chart(ctx, {
                type: 'horizontalBar',
                data: {
                    labels: _labels,
                    datasets: [{
                        label: "Rationales",
                        backgroundColor: "rgba(2,117,216,1)",
                        borderColor: "rgba(2,117,216,1)",
                        data: _data
                    }]
                },
                options: {
                    scales: {
                        xAxes: [{
                            ticks: {
                                min: 0
                            },
                            gridLines: {
                                display: true
                            }
                        }],
                        yAxes: [{
                            gridLines: {
                                display: false
                            }
                        }],
                    },
                    legend: {
                        display: false
                    }
                }
            });
        }
    });

    $.ajax({
        dataType: "json",
        url: $AJAX_RATIONALES_LABELS_URL,
        success: function (data) {
            _labels = data.labels;
            _data = data.data;
        },
        complete: function () {
            const ctx = document.getElementById("rationalesBarChart");
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: _labels,
                    datasets: [{
                        label: "Rationales",
                        backgroundColor: "rgba(2,117,216,1)",
                        borderColor: "rgba(2,117,216,1)",
                        data: _data
                    }]
                },
                options: {
                    scales: {
                        xAxes: [{
                            gridLines: {
                                display: false
                            }
                        }],
                        yAxes: [{
                            ticks: {
                                min: 0
                            },
                            gridLines: {
                                display: true
                            }
                        }],
                    },
                    legend: {
                        display: false
                    }
                }
            });
        }
    });

    $.ajax({
        dataType: "json",
        url: $AJAX_RATIONALES_USERS_URL,
        success: function (data) {
            _labels = data.labels;
            _data = data.data;
        },
        complete: function () {
            const ctx = document.getElementById("usersPieChart");
            new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: _labels,
                    datasets: [{
                        data: _data,
                        backgroundColor: ['#007bff', '#dc3545', '#ffc107', '#28a745'],
                    }],
                },
            });
        }
    });
});