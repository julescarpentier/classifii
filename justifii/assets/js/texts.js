// DataTables
import 'datatables.net-bs4';
import 'datatables.net-select-bs4'
import 'datatables.net-buttons-bs4'

(function ($) {
    "use strict"; // Start of use strict

    $(document).ready(function () {
        let table = $('#textsDataTable').DataTable({
            deferRender: true,
            stateSave: true,
            select: {
                style: 'single',
            },
            ajax: $AJAX_TEXTS_URL,
            columns: [
                {title: '#', data: 'id'},
                {title: 'File path', data: 'fpath'},
                {title: 'Label', data: 'label'},
            ],
            buttons: [
                {
                    extend: 'selectedSingle',
                    text: '<i class="fas fa-eye"></i> Show',
                    action: function (e, dt, button, config) {
                        window.location.href = dt.row({selected: true}).data()['show_url'];
                    }
                },
                {
                    extend: 'selectedSingle',
                    text: '<i class="fas fa-comment"></i> Justify',
                    action: function (e, dt, button, config) {
                        window.location.href = dt.row({selected: true}).data()['justify_url'];
                    }
                }
            ],
            initComplete: function (settings, json) {
                table.buttons().container().appendTo($('#textsDataTable_wrapper .col-md-6').eq(0));
            }
        });
    });

})(jQuery); // End of use strict